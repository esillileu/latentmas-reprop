"""Receiver carrier acquisition use case."""

import contextlib
import time
import traceback
from typing import Any

from ..domain.models import ReceiverAcquisitionMetrics, ReceiverAcquisitionRecord
from ..domain.ports.cache_port import CachePort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..domain.services.prompts.acquisition import (
    CANDIDATE_DIGITS,
    CROSS_PAIRING_POLICY,
    RECEIVER_ANSWER_PREFIX,
    RECEIVER_PROMPT,
    RECEIVER_PROMPT_TEMPLATE_VERSION,
    SENDER_PROMPT_TEMPLATE,
    SENDER_PROMPT_TEMPLATE_VERSION,
    build_receiver_messages,
    build_sender_messages,
)
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.paths.resolver import get_git_commit_hash
from .receiver_acquisition.aggregation import aggregate_receiver_records
from .receiver_acquisition.artifacts import log_receiver_artifacts
from .receiver_acquisition.runner import run_acquisition_loop
from .receiver_acquisition.sampling import (
    SecretDigitSample,
    SenderCacheBundle,
    generate_secret_digit_samples,
    pair_different_digit_sources,
)
from .receiver_acquisition.scoring import (
    score_digit_logits,
    validate_digit_candidates,
)
from .sender_latent_probe import SenderProbeResult, run_sender_latent_probe

__all__ = [
    "CANDIDATE_DIGITS",
    "CROSS_PAIRING_POLICY",
    "RECEIVER_ANSWER_PREFIX",
    "RECEIVER_PROMPT",
    "RECEIVER_PROMPT_TEMPLATE_VERSION",
    "SENDER_PROMPT_TEMPLATE",
    "SENDER_PROMPT_TEMPLATE_VERSION",
    "ReceiverAcquisitionUseCase",
    "SecretDigitSample",
    "SenderCacheBundle",
    "aggregate_receiver_records",
    "build_receiver_messages",
    "build_sender_messages",
    "generate_secret_digit_samples",
    "pair_different_digit_sources",
    "score_digit_logits",
    "validate_digit_candidates",
]


class ReceiverAcquisitionUseCase:
    """Application use case for secret digit receiver acquisition."""

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
            receiver_prompts, _, _, _ = model.prepare_chat_batch(
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
            records, hidden, context_runtime = run_acquisition_loop(
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
                self.cache_port,
                self.tracker_port,
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
            log_receiver_artifacts(
                self.cache_port,
                self.tracker_port,
                args,
                records,
                metrics,
                mapping,
                hidden,
                probe_result,
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
