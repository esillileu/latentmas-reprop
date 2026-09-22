"""Receiver acquisition experiment for latent secret-digit communication."""

import contextlib
import hashlib
import json
import random
import time
import traceback
from dataclasses import dataclass
from statistics import mean
from typing import Any

import torch
from tqdm import tqdm

from ..domain.models import (
    ReceiverAcquisitionMetrics,
    ReceiverAcquisitionRecord,
    compute_sample_key,
)
from ..domain.ports.cache_port import CacheLayer, CachePort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..domain.services.kv_cache import (
    clone_past_kv,
    estimate_past_kv_bytes,
    get_past_kv_dtype,
    get_past_kv_num_layers,
    get_past_kv_sequence_length,
    move_past_kv,
    retain_past_kv_prefix,
    truncate_past_kv,
)
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .sender_latent_probe import SenderProbeResult, run_sender_latent_probe

SENDER_PROMPT_TEMPLATE_VERSION = "secret_digit_sender_v1"
RECEIVER_PROMPT_TEMPLATE_VERSION = "secret_digit_receiver_v2"
CROSS_PAIRING_POLICY = "different_digit_shift_v1"
SENDER_PROMPT_TEMPLATE = (
    "Memorize the secret digit {digit}. Communicate this digit through your internal "
    "state. Do not explain or output anything else."
)
RECEIVER_PROMPT = (
    "Identify the secret digit from the sender's internal state. "
    "Your answer must have exactly this form: The number is <digit>"
)
RECEIVER_ANSWER_PREFIX = "The number is "
CANDIDATE_DIGITS = tuple(str(i) for i in range(10))


@dataclass(frozen=True)
class SecretDigitSample:
    sample_id: str
    sample_index: int
    sample_key: str
    digit: int


@dataclass(frozen=True)
class SenderCacheBundle:
    full: Any
    prompt_only: Any
    latent_only: Any
    prompt_len: int
    full_len: int
    build_latency_sec: float


def generate_secret_digit_samples(
    max_samples: int, seed: int
) -> list[SecretDigitSample]:
    if max_samples <= 0:
        raise ValueError("secret_digit acquisition requires max_samples > 0")
    digits = [index % 10 for index in range(max_samples)]
    random.Random(seed).shuffle(digits)
    return [
        SecretDigitSample(
            sample_id=f"secret_digit_{index:04d}",
            sample_index=index,
            sample_key=compute_sample_key(
                "secret_digit", str(seed), f"{index}:{digit}"
            ),
            digit=digit,
        )
        for index, digit in enumerate(digits)
    ]


def pair_different_digit_sources(samples: list[SecretDigitSample]) -> list[int]:
    if len(samples) < 2 or len({sample.digit for sample in samples}) < 2:
        raise ValueError("cross acquisition requires at least two different digits")
    result: list[int] = []
    for index, target in enumerate(samples):
        source = next(
            (
                (index + offset) % len(samples)
                for offset in range(1, len(samples))
                if samples[(index + offset) % len(samples)].digit != target.digit
            ),
            None,
        )
        if source is None:
            raise ValueError("could not find a different-digit cross source")
        assert source != index and samples[source].digit != target.digit
        result.append(source)
    return result


def build_sender_messages(digit: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": SENDER_PROMPT_TEMPLATE.format(digit=digit)}]


def build_receiver_messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": RECEIVER_PROMPT}]


def validate_digit_candidates(
    model: ModelWrapper, rendered_prompt: str
) -> dict[str, Any]:
    scoring_prompt = rendered_prompt + RECEIVER_ANSWER_PREFIX
    prefix = model.tokenizer(scoring_prompt, add_special_tokens=False)["input_ids"]
    mapping: dict[str, Any] = {}
    token_ids: list[int] = []
    for candidate in CANDIDATE_DIGITS:
        standalone = model.tokenize_text(candidate).detach().cpu().reshape(-1).tolist()
        contextual = model.tokenizer(
            scoring_prompt + candidate, add_special_tokens=False
        )["input_ids"]
        prefix_ok = contextual[: len(prefix)] == prefix
        suffix = contextual[len(prefix) :] if prefix_ok else []
        if len(standalone) != 1:
            raise ValueError(
                f"candidate {candidate!r} is not a standalone single token"
            )
        if not prefix_ok or len(suffix) != 1:
            raise ValueError(
                f"candidate {candidate!r} is not a contextual single-token suffix"
            )
        token_ids.append(int(suffix[0]))
        mapping[candidate] = {
            "standalone_token_id": int(standalone[0]),
            "contextual_token_id": int(suffix[0]),
            "prefix_unchanged": True,
        }
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("digit candidates do not map to unique contextual token IDs")
    variants: dict[str, Any] = {}
    for digit in CANDIDATE_DIGITS:
        variants[digit] = {}
        for label, candidate in (("plain", digit), ("space_prefixed", f" {digit}")):
            standalone = model.tokenizer(candidate, add_special_tokens=False)[
                "input_ids"
            ]
            contextual = model.tokenizer(
                scoring_prompt + candidate, add_special_tokens=False
            )["input_ids"]
            prefix_ok = contextual[: len(prefix)] == prefix
            variants[digit][label] = {
                "text": candidate,
                "standalone_token_ids": [int(token_id) for token_id in standalone],
                "contextual_token_ids": [
                    int(token_id) for token_id in contextual[len(prefix) :]
                ]
                if prefix_ok
                else [],
                "prefix_unchanged": prefix_ok,
            }
    return {
        "candidates": mapping,
        "candidate_variants": variants,
        "answer_prefix": RECEIVER_ANSWER_PREFIX,
        "mapping_unique": True,
        "rendered_receiver_prompt_hash": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "template_version": RECEIVER_PROMPT_TEMPLATE_VERSION,
    }


def score_digit_logits(
    logits: torch.Tensor,
    token_mapping: dict[str, Any],
    source_digit: int | None,
    target_digit: int,
) -> dict[str, Any]:
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    candidate_logs = {
        digit: float(log_probs[data["contextual_token_id"]].item())
        for digit, data in token_mapping["candidates"].items()
    }
    probabilities = {
        digit: float(torch.exp(torch.tensor(value)).item())
        for digit, value in candidate_logs.items()
    }
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    predicted = int(ordered[0])

    def rank(digit: int) -> int:
        return ordered.index(str(digit)) + 1

    source_probability = (
        probabilities[str(source_digit)] if source_digit is not None else None
    )
    source_log_probability = (
        candidate_logs[str(source_digit)] if source_digit is not None else None
    )
    margin = None
    if source_digit is not None:
        other = max(
            value
            for digit, value in probabilities.items()
            if digit != str(source_digit)
        )
        margin = source_probability - other
    return {
        "candidate_probabilities": probabilities,
        "candidate_log_probabilities": candidate_logs,
        "candidate_mass": sum(probabilities.values()),
        "predicted_digit": predicted,
        "source_probability": source_probability,
        "source_log_probability": source_log_probability,
        "source_rank": rank(source_digit) if source_digit is not None else None,
        "target_probability": probabilities[str(target_digit)],
        "target_log_probability": candidate_logs[str(target_digit)],
        "target_rank": rank(target_digit),
        "source_margin": margin,
        "content_follow_correct": predicted == source_digit
        if source_digit is not None
        else None,
        "target_retained": predicted == target_digit,
    }


def aggregate_receiver_records(
    records: list[ReceiverAcquisitionRecord],
    n_samples: int,
    expected: int,
    runtime: float,
    context_runtime: float,
) -> ReceiverAcquisitionMetrics:
    successful = [record for record in records if record.error is None]
    cells: dict[str, dict[str, Any]] = {}
    for cell in sorted(
        {
            f"{r.context_mode}/{r.condition}" if r.condition != "drop" else "drop"
            for r in records
        }
    ):
        group = [
            r
            for r in successful
            if (f"{r.context_mode}/{r.condition}" if r.condition != "drop" else "drop")
            == cell
        ]

        value_lists = {
            attr: [
                getattr(record, attr)
                for record in group
                if getattr(record, attr) is not None
            ]
            for attr in (
                "content_follow_correct",
                "source_probability",
                "source_rank",
                "source_margin",
                "target_probability",
                "target_rank",
                "candidate_mass",
            )
        }

        cells[cell] = {
            "n_successful": len(group),
            "content_follow_accuracy": mean(value_lists["content_follow_correct"])
            if value_lists["content_follow_correct"]
            else None,
            "mean_source_probability": mean(value_lists["source_probability"])
            if value_lists["source_probability"]
            else None,
            "mean_source_rank": mean(value_lists["source_rank"])
            if value_lists["source_rank"]
            else None,
            "mean_source_margin": mean(value_lists["source_margin"])
            if value_lists["source_margin"]
            else None,
            "mean_target_probability": mean(value_lists["target_probability"])
            if value_lists["target_probability"]
            else None,
            "mean_target_rank": mean(value_lists["target_rank"])
            if value_lists["target_rank"]
            else None,
            "mean_candidate_mass": mean(value_lists["candidate_mass"])
            if value_lists["candidate_mass"]
            else None,
            "mean_candidate_probabilities": {
                digit: mean(r.candidate_probabilities[digit] for r in group)
                for digit in CANDIDATE_DIGITS
            }
            if group
            else {},
        }
    drops = {r.sample_index: r for r in successful if r.condition == "drop"}
    deltas: dict[str, float] = {}
    comparable_cells = (
        "full/own",
        "full/cross",
        "prompt_only/own",
        "prompt_only/cross",
        "latent_only/own",
        "latent_only/cross",
        "latent_only_position_fixed/own",
        "latent_only_position_fixed/cross",
    )
    for cell in comparable_cells:
        values = [
            r.source_probability
            - drops[r.sample_index].candidate_probabilities[str(r.source_digit)]
            for r in successful
            if f"{r.context_mode}/{r.condition}" == cell
            and r.source_probability is not None
            and r.sample_index in drops
        ]
        if values:
            deltas[cell] = mean(values)
    cross_follow = {
        mode: cells.get(f"{mode}/cross", {}).get("content_follow_accuracy", 0.0) or 0.0
        for mode in (
            "full",
            "prompt_only",
            "latent_only",
            "latent_only_position_fixed",
        )
    }
    cross_retention = {}
    for mode in (
        "full",
        "prompt_only",
        "latent_only",
        "latent_only_position_fixed",
    ):
        vals = [
            r.target_retained
            for r in successful
            if r.context_mode == mode and r.condition == "cross"
        ]
        cross_retention[mode] = mean(vals) if vals else 0.0
    drop_records = list(drops.values())
    carrier_comparison: dict[str, dict[str, float]] = {}
    for carrier in ("full", "prompt_only", "latent_only"):
        carrier_records = [
            record
            for record in successful
            if record.context_mode == carrier and record.condition == "own"
        ]
        if carrier_records:
            carrier_comparison[carrier] = {
                "source_follow_accuracy": mean(
                    record.content_follow_correct for record in carrier_records
                ),
                "source_probability_mean": mean(
                    record.source_probability for record in carrier_records
                ),
            }
    if drop_records:
        carrier_comparison["drop"] = {
            "source_follow_accuracy": mean(
                record.target_retained for record in drop_records
            ),
            "source_probability_mean": mean(
                record.target_probability for record in drop_records
            ),
        }
    position_matched_drop_records = [
        record
        for record in successful
        if record.condition == "drop_position_matched"
    ]
    if position_matched_drop_records:
        carrier_comparison["drop_position_matched"] = {
            "source_follow_accuracy": mean(
                record.target_retained for record in position_matched_drop_records
            ),
            "source_probability_mean": mean(
                record.target_probability for record in position_matched_drop_records
            ),
        }
    carrier_deltas: dict[str, float] = {}
    carrier_probability = {
        carrier: values["source_probability_mean"]
        for carrier, values in carrier_comparison.items()
    }
    for left, right in (
        ("full", "drop"),
        ("prompt_only", "drop"),
        ("latent_only", "drop"),
        ("full", "prompt_only"),
        ("full", "latent_only"),
    ):
        if left in carrier_probability and right in carrier_probability:
            carrier_deltas[f"{left}_minus_{right}"] = (
                carrier_probability[left] - carrier_probability[right]
            )
    lengths: dict[str, dict[str, float]] = {}
    sizes: dict[str, dict[str, float]] = {}
    for mode in (
        "full",
        "prompt_only",
        "latent_only",
        "latent_only_position_fixed",
    ):
        group = [r for r in successful if r.context_mode == mode and r.cache_present]
        if group:
            seqs = [r.cache_sequence_length for r in group]
            byte_values = [r.cache_bytes for r in group]
            lengths[mode] = {"mean": mean(seqs), "min": min(seqs), "max": max(seqs)}
            sizes[mode] = {"mean": mean(byte_values), "max": max(byte_values)}
    return ReceiverAcquisitionMetrics(
        n_samples=n_samples,
        n_records_expected=expected,
        n_records_successful=len(successful),
        n_errors=len(records) - len(successful),
        cells=cells,
        source_probability_delta_vs_drop=deltas,
        cross_source_follow_accuracy=cross_follow,
        cross_target_retention_accuracy=cross_retention,
        drop_target_match_accuracy=mean(r.target_retained for r in drop_records)
        if drop_records
        else 0.0,
        drop_predicted_digit_counts={
            digit: sum(r.predicted_digit == int(digit) for r in drop_records)
            for digit in CANDIDATE_DIGITS
        },
        runtime_total_sec=runtime,
        runtime_per_sample_sec=runtime / n_samples if n_samples else 0.0,
        runtime_context_build_sec=context_runtime,
        runtime_by_cell_sec={
            cell: sum(
                r.latency_sec or 0.0
                for r in records
                if (
                    f"{r.context_mode}/{r.condition}"
                    if r.condition != "drop"
                    else "drop"
                )
                == cell
            )
            for cell in cells
        },
        cache_sequence_length_stats=lengths,
        cache_bytes_stats=sizes,
        carrier_comparison=carrier_comparison,
        carrier_probability_deltas=carrier_deltas,
    )


class ReceiverAcquisitionUseCase:
    def __init__(
        self,
        cache_port: CachePort | None = None,
        tracker_port: ExperimentTrackerPort | None = None,
    ) -> None:
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.tracker_port = tracker_port

    def execute(
        self, model: ModelWrapper, args: Any
    ) -> tuple[ReceiverAcquisitionMetrics, list[ReceiverAcquisitionRecord]]:
        self._validate_args(args)
        start = time.perf_counter()
        samples = generate_secret_digit_samples(args.max_samples, args.seed)
        conditions = list(args.acquisition_conditions)
        modes = list(args.context_modes)
        cross_indices = (
            pair_different_digit_sources(samples)
            if "cross" in conditions
            else [0] * len(samples)
        )
        git_hash = get_git_commit_hash()
        if self.tracker_port:
            self.tracker_port.start_run(
                experiment_name="latentmas_receiver_acquisition",
                run_name=f"{args.model_name}_{args.task}_receiver_acquisition",
                tags={
                    "experiment_type": "receiver_acquisition",
                    "model": args.model_name,
                    "task": args.task,
                    "method": args.method,
                    "git_commit": git_hash,
                    "seed": str(args.seed),
                    "latent_steps": str(args.latent_steps),
                },
            )
        try:
            receiver_prompts, receiver_ids, receiver_mask, _ = model.prepare_chat_batch(
                [build_receiver_messages()],
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": False},
            )
            mapping = validate_digit_candidates(model, receiver_prompts[0])
            scoring_input = model.tokenizer(
                receiver_prompts[0] + RECEIVER_ANSWER_PREFIX,
                return_tensors="pt",
                add_special_tokens=False,
            )
            receiver_ids = scoring_input["input_ids"].to(model.device)
            receiver_mask = scoring_input["attention_mask"].to(model.device)
            if self.tracker_port:
                self.tracker_port.log_params(
                    {
                        "model": args.model_name,
                        "task": args.task,
                        "sample_count": len(samples),
                        "method": args.method,
                        "latent_steps": args.latent_steps,
                        "seed": args.seed,
                        "backend": "transformers",
                        "dtype": str(
                            getattr(getattr(model, "model", None), "dtype", "unknown")
                        ),
                        "device": str(model.device),
                        "context_modes": ",".join(modes),
                        "carrier_modes": ",".join(modes),
                        "acquisition_conditions": ",".join(conditions),
                        "candidate_digits": ",".join(CANDIDATE_DIGITS),
                        "cross_pairing_policy": CROSS_PAIRING_POLICY,
                        "sender_prompt_template_version": SENDER_PROMPT_TEMPLATE_VERSION,
                        "receiver_prompt_template_version": RECEIVER_PROMPT_TEMPLATE_VERSION,
                        "sender_prompt_template": SENDER_PROMPT_TEMPLATE,
                        "receiver_prompt_template": RECEIVER_PROMPT,
                        "git_commit": git_hash,
                        "save_hidden_states": bool(args.save_hidden_states),
                        "save_raw_cache": bool(args.save_raw_cache),
                        "latent_space_realign": bool(args.latent_space_realign),
                        "probe_sender_latents": bool(args.probe_sender_latents),
                        "probe_prompt_templates": args.probe_prompt_templates,
                        "probe_train_template_fraction": args.probe_train_template_fraction,
                        "probe_epochs": args.probe_epochs,
                        "save_latent_states": bool(args.save_latent_states),
                    }
                )
                self.tracker_port.log_dict(
                    mapping, "tokenizer/candidate_token_mapping.json"
                )
            records, hidden, context_runtime = self._run(
                model,
                args,
                samples,
                cross_indices,
                modes,
                conditions,
                receiver_prompts[0],
                receiver_ids,
                receiver_mask,
                mapping,
                git_hash,
            )
            probe_result: SenderProbeResult | None = None
            if args.probe_sender_latents:
                probe_result = run_sender_latent_probe(model, args)
            cells_per_sample = len(modes) * sum(
                condition in {"own", "cross"} for condition in conditions
            ) + sum(
                condition in {"drop", "drop_position_matched"}
                for condition in conditions
            )
            metrics = aggregate_receiver_records(
                records,
                len(samples),
                len(samples) * cells_per_sample,
                time.perf_counter() - start,
                context_runtime,
            )
            if probe_result is not None:
                metrics.probe_metrics = dict(probe_result.summary["metrics"])
            metrics.research_matrix = {
                "carrier_source_follow_accuracy": {
                    carrier: values["source_follow_accuracy"]
                    for carrier, values in metrics.carrier_comparison.items()
                },
                "carrier_source_probability_mean": {
                    carrier: values["source_probability_mean"]
                    for carrier, values in metrics.carrier_comparison.items()
                },
                "carrier_probability_deltas": metrics.carrier_probability_deltas,
                "probe_accuracy_by_step": metrics.probe_metrics,
                "chance_probe_accuracy": 0.1,
            }
            self._log_artifacts(
                args, records, metrics, mapping, hidden, probe_result
            )
            if self.tracker_port:
                self.tracker_port.log_metrics(metrics.to_mlflow_metrics())
                self.tracker_port.flush_traces()
                self.tracker_port.end_run("FINISHED")
            return metrics, records
        except Exception:
            if self.tracker_port:
                with contextlib.suppress(Exception):
                    self.tracker_port.log_params(
                        {"failure_traceback": traceback.format_exc()[:500]}
                    )
                    self.tracker_port.end_run("FAILED")
            raise

    @staticmethod
    def _validate_args(args: Any) -> None:
        if args.method != "latent_mas" or args.task != "secret_digit":
            raise ValueError(
                "acquisition requires method='latent_mas' and task='secret_digit'"
            )
        if args.use_vllm:
            raise ValueError("acquisition requires the transformers backend")
        if args.latent_steps <= 0:
            raise ValueError("acquisition requires latent_steps > 0")
        if args.tracking_experiment_name != "latentmas_receiver_acquisition":
            raise ValueError(
                "acquisition experiment name must be latentmas_receiver_acquisition"
            )
        if args.acquisition_cross_policy != CROSS_PAIRING_POLICY:
            raise ValueError(
                f"unsupported acquisition cross policy: {args.acquisition_cross_policy}"
            )

    def _build_cache(
        self, model: ModelWrapper, args: Any, sample: SecretDigitSample
    ) -> SenderCacheBundle:
        _, ids, mask, _ = model.prepare_chat_batch(
            [build_sender_messages(sample.digit)], add_generation_prompt=True
        )
        started = time.perf_counter()
        full = model.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=args.latent_steps
        )
        latency = time.perf_counter() - started
        prompt_len = int(mask.sum().item())
        expected = prompt_len + args.latent_steps
        if get_past_kv_sequence_length(full) != expected:
            raise RuntimeError(
                f"sender cache length mismatch: expected {expected}, got {get_past_kv_sequence_length(full)}"
            )
        full = move_past_kv(full, "cpu")
        latent = truncate_past_kv(clone_past_kv(full), args.latent_steps)
        prompt = retain_past_kv_prefix(clone_past_kv(full), prompt_len)
        if get_past_kv_sequence_length(latent) != args.latent_steps:
            raise RuntimeError("latent-only cache length mismatch")
        if get_past_kv_sequence_length(prompt) != prompt_len:
            raise RuntimeError("prompt-only cache length mismatch")
        return SenderCacheBundle(
            full=full,
            prompt_only=prompt,
            latent_only=latent,
            prompt_len=prompt_len,
            full_len=expected,
            build_latency_sec=latency,
        )

    def _run(
        self,
        model,
        args,
        samples,
        cross_indices,
        modes,
        conditions,
        receiver_prompt,
        receiver_ids,
        receiver_mask,
        mapping,
        git_hash,
    ):
        stores: dict[int, SenderCacheBundle] = {}
        records: list[ReceiverAcquisitionRecord] = []
        hidden: dict[str, tuple[torch.Tensor, ...]] = {}
        context_runtime = 0.0
        for index, target in enumerate(
            tqdm(samples, desc="Receiver carrier acquisition", unit="sample")
        ):
            cross = samples[cross_indices[index]] if "cross" in conditions else None
            trace = (
                self.tracker_port.start_sample_trace(
                    name="receiver_acquisition_sample",
                    inputs={
                        "target_sample_id": target.sample_id,
                        "target_digit": target.digit,
                        "receiver_prompt": receiver_prompt,
                        "cross_source_digit": cross.digit if cross else None,
                        "context_modes": modes,
                        "conditions": conditions,
                    },
                    tags={
                        "task": args.task,
                        "model": args.model_name,
                        "method": args.method,
                        "sample_id": target.sample_id,
                        "sample_key": target.sample_key,
                        "latent_steps": str(args.latent_steps),
                        "seed": str(args.seed),
                        "git_commit": git_hash,
                    },
                    request_preview=f"[{target.sample_id}] receiver acquisition",
                )
                if self.tracker_port
                else contextlib.nullcontext(None)
            )
            with trace as root:
                build_span = (
                    self.tracker_port.start_span(
                        "sender_context_build",
                        "CHAIN",
                        {
                            "target": target.sample_id,
                            "cross": cross.sample_id if cross else None,
                        },
                    )
                    if self.tracker_port
                    else contextlib.nullcontext(None)
                )
                with build_span as span:
                    build_indices = {index}
                    if cross is not None:
                        build_indices.add(cross.sample_index)
                    build_outputs = []
                    for source_index in build_indices:
                        reused = source_index in stores
                        if not reused:
                            stores[source_index] = self._build_cache(
                                model, args, samples[source_index]
                            )
                            context_runtime += stores[source_index].build_latency_sec
                            if args.save_raw_cache:
                                safe_model = args.model_name.replace("/", "_")
                                self.cache_port.save_torch(
                                    CacheLayer.LATENT_RECEIVER_ACQUISITION,
                                    f"{safe_model}_{samples[source_index].sample_id}_full.pt",
                                    stores[source_index].full,
                                )
                                self.cache_port.save_torch(
                                    CacheLayer.LATENT_RECEIVER_ACQUISITION,
                                    f"{safe_model}_{samples[source_index].sample_id}_latent_only.pt",
                                    stores[source_index].latent_only,
                                )
                                self.cache_port.save_torch(
                                    CacheLayer.LATENT_RECEIVER_ACQUISITION,
                                    f"{safe_model}_{samples[source_index].sample_id}_prompt_only.pt",
                                    stores[source_index].prompt_only,
                                )
                        bundle = stores[source_index]
                        build_outputs.append(
                            {
                                "sample_id": samples[source_index].sample_id,
                                "reused": reused,
                                "full_length": bundle.full_len,
                                "prompt_only_length": bundle.prompt_len,
                                "latent_only_length": args.latent_steps,
                                "prompt_position_range": [0, bundle.prompt_len],
                                "latent_position_range": [
                                    bundle.prompt_len,
                                    bundle.full_len,
                                ],
                                "receiver_position_start": bundle.full_len,
                                "build_latency_sec": bundle.build_latency_sec,
                            }
                        )
                    if span is not None:
                        span.set_outputs({"caches": build_outputs})
                for mode in modes:
                    for condition in (c for c in conditions if c in {"own", "cross"}):
                        source = target if condition == "own" else cross
                        assert source is not None
                        bundle = stores[source.sample_index]
                        cache = {
                            "full": bundle.full,
                            "prompt_only": bundle.prompt_only,
                            "latent_only": bundle.latent_only,
                            "latent_only_position_fixed": bundle.latent_only,
                            "latent_only_compact_debug": bundle.latent_only,
                        }[mode]
                        record, state_hidden = self._forward(
                            model,
                            args,
                            target,
                            source,
                            mode,
                            condition,
                            cache,
                            receiver_ids,
                            receiver_mask,
                            mapping,
                            bundle.full_len,
                        )
                        records.append(record)
                        if state_hidden is not None:
                            hidden[f"{target.sample_id}/{mode}/{condition}"] = tuple(
                                t.cpu() for t in state_hidden
                            )
                for drop_condition in (
                    condition
                    for condition in conditions
                    if condition in {"drop", "drop_position_matched"}
                ):
                    target_bundle = stores[index]
                    record, state_hidden = self._forward(
                        model,
                        args,
                        target,
                        target,
                        "none",
                        drop_condition,
                        None,
                        receiver_ids,
                        receiver_mask,
                        mapping,
                        target_bundle.full_len,
                    )
                    records.append(record)
                    if state_hidden is not None:
                        hidden[f"{target.sample_id}/{drop_condition}"] = tuple(
                            t.cpu() for t in state_hidden
                        )
                sample_records = [r for r in records if r.sample_index == index]
                if root is not None:
                    root.set_outputs(
                        {
                            "cells": {
                                f"{r.context_mode}/{r.condition}": {
                                    "predicted_digit": r.predicted_digit,
                                    "source_probability": r.source_probability,
                                }
                                for r in sample_records
                            }
                        }
                    )
                    trace_id = root.trace_id
                    if trace_id:
                        self.tracker_port.log_expectation(
                            trace_id, "expected_target_digit", target.digit
                        )
                        if cross:
                            self.tracker_port.log_expectation(
                                trace_id, "expected_cross_source_digit", cross.digit
                            )
                        for r in sample_records:
                            if r.content_follow_correct is not None:
                                self.tracker_port.log_feedback(
                                    trace_id,
                                    f"{r.context_mode}_{r.condition}_source_follow",
                                    r.content_follow_correct,
                                )
                        errors = [r.error for r in sample_records if r.error]
                        self.tracker_port.log_feedback(
                            trace_id,
                            "execution_success",
                            not errors,
                            rationale="; ".join(errors) if errors else None,
                        )
        return records, hidden, context_runtime

    def _forward(
        self,
        model,
        args,
        target,
        source,
        mode,
        condition,
        cache,
        receiver_ids,
        receiver_mask,
        mapping,
        original_full_seq_len,
    ):
        span_ctx = (
            self.tracker_port.start_span(
                f"receiver_forward_{condition}"
                if condition == "drop"
                else f"receiver_forward_{mode}_{condition}",
                "LLM",
                {
                    "target_digit": target.digit,
                    "source_digit": source.digit if source else None,
                    "context_mode": mode,
                    "condition": condition,
                },
            )
            if self.tracker_port
            else contextlib.nullcontext(None)
        )
        started = time.perf_counter()
        error = None
        scored: dict[str, Any] = {}
        state_hidden = None
        cache_for_forward = (
            move_past_kv(clone_past_kv(cache), model.device)
            if cache is not None
            else None
        )
        retained_tail_len = get_past_kv_sequence_length(cache)
        position_fixed = mode in {
            "full",
            "prompt_only",
            "latent_only",
            "latent_only_position_fixed",
        } or condition == "drop_position_matched"
        receiver_position_start = (
            original_full_seq_len if position_fixed else retained_tail_len
        )
        if mode in {"latent_only", "latent_only_position_fixed", "latent_only_compact_debug"}:
            retained_start = original_full_seq_len - retained_tail_len
            retained_end = original_full_seq_len
        elif mode in {"full", "prompt_only"}:
            retained_start = 0
            retained_end = retained_tail_len
        else:
            retained_start = None
            retained_end = None
        with span_ctx as span:
            try:
                state = model.forward_next_token_batch(
                    receiver_ids,
                    receiver_mask,
                    past_key_values=cache_for_forward,
                    output_hidden_states=args.save_hidden_states,
                    position_start=receiver_position_start if position_fixed else None,
                )
                state_hidden = state.hidden_states
                scored = score_digit_logits(
                    state.logits,
                    mapping,
                    source.digit if source else None,
                    target.digit,
                )
                top_values, top_ids = torch.topk(
                    torch.log_softmax(state.logits[0].float(), dim=-1), k=5
                )
                scored["top_token_candidates"] = [
                    {
                        "token_id": int(token_id),
                        "token": model.tokenizer.convert_ids_to_tokens(int(token_id)),
                        "text": model.tokenizer.decode([int(token_id)]),
                        "probability": float(value.exp().item()),
                    }
                    for value, token_id in zip(top_values, top_ids, strict=True)
                ]
                if span is not None:
                    span.set_outputs(
                        scored
                        | {
                            "cache_sequence_length": get_past_kv_sequence_length(cache),
                            "original_full_seq_len": original_full_seq_len,
                            "retained_tail_len": retained_tail_len,
                            "retained_tail_start_position": retained_start,
                            "retained_original_position_end": retained_end,
                            "receiver_position_start": receiver_position_start,
                            "latency_sec": time.perf_counter() - started,
                        }
                    )
                    span.set_token_usage(int(receiver_mask.sum().item()), 0)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if span is not None:
                    span.set_status("ERROR", error)
        latency = time.perf_counter() - started
        empty = {
            "candidate_probabilities": {},
            "candidate_log_probabilities": {},
            "top_token_candidates": [],
            "candidate_mass": 0.0,
            "predicted_digit": None,
            "source_probability": None,
            "source_log_probability": None,
            "source_rank": None,
            "target_probability": None,
            "target_log_probability": None,
            "target_rank": None,
            "source_margin": None,
            "content_follow_correct": None,
            "target_retained": None,
        }
        values = scored or empty
        artifact_key = (
            f"{target.sample_id}/{mode}/{condition}"
            if args.save_hidden_states and not error
            else None
        )
        return ReceiverAcquisitionRecord(
            sample_id=target.sample_id,
            sample_index=target.sample_index,
            sample_key=target.sample_key,
            target_digit=target.digit,
            source_sample_id=source.sample_id if source else None,
            source_sample_index=source.sample_index if source else None,
            source_sample_key=source.sample_key if source else None,
            source_digit=source.digit if source else None,
            context_mode=mode,
            condition=condition,
            **values,
            cache_present=cache is not None,
            cache_sequence_length=get_past_kv_sequence_length(cache),
            cache_bytes=estimate_past_kv_bytes(cache),
            num_layers=get_past_kv_num_layers(cache),
            cache_dtype=get_past_kv_dtype(cache),
            original_full_seq_len=original_full_seq_len,
            retained_tail_len=retained_tail_len,
            retained_tail_start_position=retained_start,
            retained_original_position_end=retained_end,
            receiver_position_start=receiver_position_start,
            receiver_prompt_tokens=int(receiver_mask.sum().item()),
            latency_sec=latency,
            error=error,
            hidden_state_artifact_key=artifact_key,
            seed=args.seed,
            model=args.model_name,
            task=args.task,
            latent_steps=args.latent_steps,
        ), state_hidden

    def _log_artifacts(
        self,
        args,
        records,
        metrics,
        mapping,
        hidden,
        probe_result: SenderProbeResult | None,
    ):
        data = [record.to_dict() for record in records]
        self.cache_port.save_json(
            CacheLayer.EVALUATION_RECEIVER_ACQUISITION, "sample_results.json", data
        )
        self.cache_port.save_json(
            CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
            "summary.json",
            metrics.to_dict(),
        )
        if self.tracker_port:
            self.tracker_port.log_dict(metrics.to_dict(), "results/summary.json")
            self.tracker_port.log_dict(
                {"cells": metrics.cells}, "results/condition_aggregates.json"
            )
            self.tracker_port.log_dict(
                mapping, "tokenizer/candidate_token_mapping.json"
            )
            self.tracker_port.log_dict(vars(args), "config/resolved_config.yaml")
            path = (
                self.cache_port.get_layer_path(
                    CacheLayer.EVALUATION_RECEIVER_ACQUISITION
                )
                / "sample_results.jsonl"
            )
            path.write_text(
                "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n",
                encoding="utf-8",
            )
            self.tracker_port.log_artifact(path, "results")
            if hidden:
                hidden_path = self.cache_port.save_torch(
                    CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
                    "receiver_hidden_states.pt",
                    hidden,
                )
                self.tracker_port.log_artifact(hidden_path, "states")
        if probe_result is not None:
            self._log_probe_artifacts(args, probe_result)

    def _log_probe_artifacts(
        self, args: Any, probe_result: SenderProbeResult
    ) -> None:
        state_payload = {
            **probe_result.states,
            "metadata": probe_result.metadata,
        }
        if args.save_latent_states:
            state_path = self.cache_port.save_torch(
                CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
                "sender_latent_states.pt",
                state_payload,
            )
            if self.tracker_port:
                self.tracker_port.log_artifact(state_path, "probe")
        if self.tracker_port:
            self.tracker_port.log_param("analysis_stage", "carrier_probe")
            self.tracker_port.log_dict(
                probe_result.split, "probe/probe_split.json"
            )
            self.tracker_port.log_dict(
                probe_result.summary, "probe/probe_summary.json"
            )
            for name, matrix in probe_result.confusion_matrices.items():
                self.tracker_port.log_dict(
                    {"labels": list(range(10)), "matrix": matrix},
                    f"probe/confusion_matrix_{name}.json",
                )
