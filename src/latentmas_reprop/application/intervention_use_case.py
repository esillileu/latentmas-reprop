"""Latent communication intervention experiment use case.

Evaluates four conditions:
    own:  q_i + H_i                 (own latent context)
    cross:q_i + H_j (j != i)        (foreign latent context)
    drop: q_i + no past_key_values  (communication dropped)
    zero: q_i + zero-filled cache   (shape/position preserved, content zeroed)
"""

import contextlib
import json
import random
import sys
import time
import traceback
from typing import Any

import torch
import yaml
from tqdm import tqdm

from ..domain.models import (
    InterventionCondition,
    InterventionMetrics,
    SampleInterventionRecord,
    compute_sample_key,
)
from ..domain.ports.cache_port import CacheLayer, CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..domain.services.latent_mas import (
    LatentMASMethod,
    clone_past_kv,
    create_zero_past_kv,
    move_past_kv,
)
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash


def generate_cross_indices(
    n: int,
    policy: str = "shift_1",
    seed: int = 42,
    cache_seq_lens: list[int] | None = None,
) -> list[int]:
    """Generate deterministic permutation indices for cross-sample pairing.

    Policies:
        shift_1:       i -> (i+1) % n
        derangement:   random derangement seeded by `seed`
        length_matched: minimize |len(H_i) - len(H_j)| via optimal assignment

    Guarantees that index[i] != i for all i (derangement property).
    """
    if n < 2:
        raise ValueError(
            f"Cross-sample intervention requires at least 2 samples, got {n}."
        )

    if policy == "shift_1":
        return [(i + 1) % n for i in range(n)]

    if policy == "derangement":
        rng = random.Random(seed)
        indices = list(range(n))
        for _ in range(1000):
            perm = list(indices)
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm
        # Fallback
        return [(i + 1) % n for i in range(n)]

    if policy == "length_matched":
        if cache_seq_lens is None:
            return [(i + 1) % n for i in range(n)]
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            lens = cache_seq_lens
            cost = np.zeros((n, n))
            for ii in range(n):
                for jj in range(n):
                    cost[ii, jj] = 1e9 if ii == jj else abs(lens[ii] - lens[jj])
            _, cols = linear_sum_assignment(cost)
            result = cols.tolist()
            # Verify derangement - fallback if violated (shouldn't happen for n>=2)
            if all(result[i] != i for i in range(n)):
                return result
        except ImportError:
            pass
        return [(i + 1) % n for i in range(n)]

    # Default
    return [(i + 1) % n for i in range(n)]


def compute_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Compute McNemar test statistic with continuity correction and exact binomial p-value."""
    n = b + c
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b_own_correct_other_wrong": b,
            "c_own_wrong_other_correct": c,
            "discordant_pairs": 0,
        }
    diff = abs(b - c)
    stat = ((diff - 1) ** 2) / n if diff >= 1 else 0.0
    try:
        from scipy.stats import binomtest

        p_val = binomtest(k=b, n=n, p=0.5).pvalue
    except Exception:
        p_val = 1.0
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 4),
        "b_own_correct_other_wrong": b,
        "c_own_wrong_other_correct": c,
        "discordant_pairs": n,
    }


def get_kv_sequence_length(past_kv: Any) -> int:
    """Safely obtain sequence length from past KV cache structure."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return past_kv.get_seq_length()
    if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
        first_layer = past_kv[0]
        if isinstance(first_layer, (list, tuple)) and len(first_layer) > 0:
            tensor = first_layer[0]
            if hasattr(tensor, "shape") and len(tensor.shape) >= 2:
                return int(tensor.shape[-2])
    return 0


def get_kv_num_layers(past_kv: Any) -> int:
    """Return number of KV cache layers."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "layers"):
        return len(past_kv.layers)
    if isinstance(past_kv, (list, tuple)):
        return len(past_kv)
    return 0


def get_kv_dtype(past_kv: Any) -> str | None:
    """Return string dtype of first KV tensor, or None."""
    if past_kv is None:
        return None
    if hasattr(past_kv, "layers") and len(past_kv.layers) > 0:
        layer = past_kv.layers[0]
        if hasattr(layer, "keys") and torch.is_tensor(layer.keys):
            return str(layer.keys.dtype)
    if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
        first = past_kv[0]
        if isinstance(first, (list, tuple)) and len(first) > 0:
            t = first[0]
            if torch.is_tensor(t):
                return str(t.dtype)
    return None


def estimate_cache_bytes(past_kv: Any) -> int:
    """Estimate total bytes in KV cache tensors (CPU-side)."""
    if past_kv is None:
        return 0
    total = 0
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    total += t.nelement() * t.element_size()
        return total
    if isinstance(past_kv, (list, tuple)):
        for layer in past_kv:
            if isinstance(layer, (list, tuple)):
                for t in layer:
                    if torch.is_tensor(t):
                        total += t.nelement() * t.element_size()
            elif torch.is_tensor(layer):
                total += layer.nelement() * layer.element_size()
    return total


class InterventionUseCase:
    """Application use case for latent communication intervention experiments.

    Evaluates conditions: own, cross, drop, zero.
    Conditions are generic - any subset may be requested via args.intervention_conditions.

    own:  q_i + H_i
    cross:q_i + H_j (j != i)
    drop: q_i + no past_key_values (communication dropped)
    zero: q_i + zero-filled cache matching shape of H_i
    """

    def __init__(
        self,
        dataset_port: DatasetPort | None = None,
        cache_port: CachePort | None = None,
        evaluator_port: EvaluatorPort | None = None,
        tracker_port: ExperimentTrackerPort | None = None,
    ) -> None:
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR
        self.tracker_port = tracker_port

    def execute(
        self,
        model: ModelWrapper,
        args: Any,
    ) -> tuple[InterventionMetrics, list[SampleInterventionRecord]]:
        if getattr(args, "method", "") != "latent_mas":
            raise ValueError(
                f"Intervention harness requires method='latent_mas', got '{getattr(args, 'method', '')}'"
            )

        start_total_time = time.time()

        # Parse conditions
        raw_conditions = getattr(
            args, "intervention_conditions", ["own", "cross", "drop", "zero"]
        )
        if isinstance(raw_conditions, str):
            conditions = [c.strip() for c in raw_conditions.split(",") if c.strip()]
        else:
            conditions = list(raw_conditions)

        cross_policy = getattr(args, "cross_policy", "shift_1")
        save_raw_cache = bool(getattr(args, "save_raw_cache", False))

        # ── 1. Load dataset ────────────────────────────────────────────────────
        dataset_iter = list(self.dataset_port.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            dataset_iter = dataset_iter[: args.max_samples]
        n_samples = len(dataset_iter)

        if "cross" in conditions and n_samples < 2:
            raise ValueError(
                f"Intervention condition 'cross' requires at least 2 samples, got {n_samples}."
            )

        # ── 2. Compute stable sample keys ─────────────────────────────────────
        split_str = getattr(args, "split", "test")
        task_str = getattr(args, "task", "gsm8k")
        sample_keys: list[str] = [
            compute_sample_key(task_str, split_str, item.get("question", ""))
            for item in dataset_iter
        ]

        # ── 3. Build method ────────────────────────────────────────────────────
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            generate_bs=1,
            args=args,
            evaluator=self.evaluator_port,
        )

        git_hash = get_git_commit_hash()
        backend_name = "vllm" if getattr(args, "use_vllm", False) else "transformers"
        model_dtype = (
            str(getattr(model.model, "dtype", "unknown"))
            if hasattr(model, "model")
            else "unknown"
        )

        # ── 4. Start MLflow run BEFORE inference ───────────────────────────────
        if self.tracker_port:
            exp_name = getattr(
                args, "tracking_experiment_name", "latentmas_intervention"
            )
            run_name = f"{args.model_name}_{args.task}_intervention"
            self.tracker_port.start_run(
                experiment_name=exp_name,
                run_name=run_name,
                tags={
                    "model": args.model_name,
                    "task": args.task,
                    "method": args.method,
                    "experiment_type": "intervention",
                    "git_commit": git_hash,
                },
            )
            self.tracker_port.log_params(
                {
                    "model": args.model_name,
                    "task": args.task,
                    "split": split_str,
                    "sample_count": n_samples,
                    "method": args.method,
                    "prompt": getattr(args, "prompt", "sequential"),
                    "latent_steps": args.latent_steps,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                    "seed": args.seed,
                    "backend": backend_name,
                    "dtype": model_dtype,
                    "cross_pairing_policy": cross_policy,
                    "git_commit": git_hash,
                    "intervention_conditions": str(conditions),
                    "device": str(model.device),
                }
            )

        print(f"\n[Intervention] Starting pilot on {n_samples} samples...")
        print(f"Conditions: {conditions} | Cross policy: {cross_policy}")

        try:
            metrics, sample_records = self._run_experiment(
                model=model,
                args=args,
                conditions=conditions,
                cross_policy=cross_policy,
                save_raw_cache=save_raw_cache,
                dataset_iter=dataset_iter,
                n_samples=n_samples,
                sample_keys=sample_keys,
                split_str=split_str,
                task_str=task_str,
                method=method,
                start_total_time=start_total_time,
                git_hash=git_hash,
            )
        except Exception:
            # Log failure and re-raise so caller can handle
            exc_text = traceback.format_exc()
            print(f"[Intervention] FAILED: {exc_text}", file=sys.stderr)
            if self.tracker_port:
                with contextlib.suppress(Exception):
                    self.tracker_port.log_params({"failure_traceback": exc_text[:500]})
                with contextlib.suppress(Exception):
                    self.tracker_port.end_run(status="FAILED")
            raise

        return metrics, sample_records

    def _run_experiment(
        self,
        *,
        model: ModelWrapper,
        args: Any,
        conditions: list[str],
        cross_policy: str,
        save_raw_cache: bool,
        dataset_iter: list[dict],
        n_samples: int,
        sample_keys: list[str],
        split_str: str,
        task_str: str,
        method: LatentMASMethod,
        start_total_time: float,
        git_hash: str,
    ) -> tuple[InterventionMetrics, list[SampleInterventionRecord]]:
        # ── Phase 1: Build latent contexts ─────────────────────────────────────
        print("\n--- Phase 1: Building latent communication contexts ---")
        contexts: list[Any] = []
        traces_list: list[list[dict]] = []
        context_build_latencies: list[float] = []
        context_seq_lens: list[int] = []
        cache_bytes_list: list[int] = []

        for idx, item in enumerate(tqdm(dataset_iter, desc="Building latent contexts")):
            t_start = time.time()
            past_kv, agent_traces = method.build_latent_contexts([item])
            t_build = time.time() - t_start

            past_kv_cpu = move_past_kv(past_kv, "cpu")
            contexts.append(past_kv_cpu)
            traces_list.append(agent_traces[0] if agent_traces else [])
            context_build_latencies.append(t_build)
            context_seq_lens.append(get_kv_sequence_length(past_kv_cpu))
            cache_bytes_list.append(estimate_cache_bytes(past_kv_cpu))

            if save_raw_cache:
                cache_name = f"latent_{args.task}_{idx}_s{args.seed}.pt"
                with contextlib.suppress(Exception):
                    self.cache_port.save_torch(
                        CacheLayer.LATENT_INTERVENTIONS, cache_name, past_kv_cpu
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Build cross indices after contexts are ready (enables length_matched)
        cross_indices: list[int] = (
            generate_cross_indices(
                n_samples,
                policy=cross_policy,
                seed=args.seed,
                cache_seq_lens=context_seq_lens
                if cross_policy == "length_matched"
                else None,
            )
            if "cross" in conditions
            else []
        )

        # ── Phase 2: Decode under all conditions ───────────────────────────────
        print("\n--- Phase 2: Decoding under intervention conditions ---")
        sample_records: list[SampleInterventionRecord] = []
        results_by_sample_and_cond: dict[int, dict[str, SampleInterventionRecord]] = {
            i: {} for i in range(n_samples)
        }
        condition_latencies: dict[str, list[float]] = {c: [] for c in conditions}
        condition_tokens: dict[str, list[int]] = {c: [] for c in conditions}

        # Cache metadata per layer for first sample
        _num_layers = get_kv_num_layers(contexts[0]) if contexts else 0
        _cache_dtype = get_kv_dtype(contexts[0]) if contexts else None

        for i, item in enumerate(tqdm(dataset_iter, desc="Intervention decoding")):
            sample_id = f"sample_{i}"
            sample_index = i
            s_key = sample_keys[i]
            gold = str(item.get("gold", ""))
            question = item.get("question", "")

            cross_j = cross_indices[i] if "cross" in conditions else None
            cross_source_sample_id = (
                f"sample_{cross_j}" if cross_j is not None else None
            )
            cross_source_key = sample_keys[cross_j] if cross_j is not None else None

            target_seq_len = context_seq_lens[i]

            # Collect for root trace outputs
            cond_preds: dict[str, tuple[str | None, bool, str]] = {}

            # Root MLflow trace
            trace_ctx = (
                self.tracker_port.start_sample_trace(
                    name="latentmas_sample",
                    inputs={
                        "sample_index": sample_index,
                        "sample_key": s_key,
                        "question": question,
                        "gold": gold,
                    },
                    tags={
                        "task": args.task,
                        "model": args.model_name,
                        "method": args.method,
                        "sample_id": sample_id,
                        "sample_key": s_key,
                        "pilot": "true",
                        "latent_steps": str(args.latent_steps),
                        "seed": str(args.seed),
                        "git_commit": git_hash,
                    },
                    request_preview=f"[{sample_id}] {question[:90]}",
                )
                if self.tracker_port
                else contextlib.nullcontext(None)
            )

            with trace_ctx as root_span:
                # Child span: build_own_context
                build_span_ctx = (
                    self.tracker_port.start_span(
                        name="build_own_context",
                        span_type="CHAIN",
                        inputs={
                            "sample_id": sample_id,
                            "sample_key": s_key,
                            "question": question,
                            "model": args.model_name,
                            "latent_steps": args.latent_steps,
                        },
                    )
                    if self.tracker_port
                    else contextlib.nullcontext(None)
                )
                with build_span_ctx as b_span:
                    own_ctx_cloned = clone_past_kv(contexts[i])
                    own_context_gpu = move_past_kv(own_ctx_cloned, model.device)
                    own_traces = [traces_list[i]]

                    agent_prompts = [
                        {
                            "name": a.get("name"),
                            "role": a.get("role"),
                            "input": a.get("input"),
                            "latent_steps": a.get("latent_steps"),
                        }
                        for a in traces_list[i]
                    ]

                    if b_span is not None:
                        b_span.set_outputs(
                            {
                                "cache_present": True,
                                "sample_key": s_key,
                                "cache_sequence_length": target_seq_len,
                                "cache_bytes": cache_bytes_list[i],
                                "num_layers": _num_layers,
                                "cache_dtype": _cache_dtype,
                                "build_latency_sec": round(
                                    context_build_latencies[i], 4
                                ),
                                "agent_prompts": agent_prompts,
                            }
                        )

                # Decode all conditions
                for cond in conditions:
                    sample_seed = args.seed + i
                    random.seed(sample_seed)
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)

                    source_sample_id: str | int | None = None
                    source_sample_index: int | None = None
                    source_s_key: str | None = None
                    chosen_context: Any = None
                    chosen_traces: list[list[dict]] | None = None
                    cache_present = False
                    src_cache_seq_len = 0

                    if cond == InterventionCondition.OWN:
                        source_sample_id = sample_id
                        source_sample_index = sample_index
                        source_s_key = s_key
                        chosen_context = own_context_gpu
                        chosen_traces = own_traces
                        cache_present = True
                        src_cache_seq_len = target_seq_len

                    elif cond == InterventionCondition.CROSS:
                        source_sample_id = cross_source_sample_id
                        source_sample_index = cross_j
                        source_s_key = cross_source_key
                        ctx_cross_cloned = clone_past_kv(contexts[cross_j])
                        chosen_context = move_past_kv(ctx_cross_cloned, model.device)
                        chosen_traces = None
                        cache_present = True
                        src_cache_seq_len = context_seq_lens[cross_j]

                    elif cond == InterventionCondition.DROP:
                        # Drop: no past_key_values at all
                        source_sample_id = None
                        source_sample_index = None
                        source_s_key = None
                        chosen_context = None
                        chosen_traces = None
                        cache_present = False
                        src_cache_seq_len = 0

                    elif cond == InterventionCondition.ZERO:
                        # Zero: same shape as H_i, but all values set to 0
                        source_sample_id = None
                        source_sample_index = None
                        source_s_key = None
                        ctx_zero = create_zero_past_kv(contexts[i])
                        chosen_context = move_past_kv(ctx_zero, model.device)
                        chosen_traces = None
                        cache_present = True
                        src_cache_seq_len = target_seq_len  # same shape as own

                    else:
                        # Unknown condition - skip gracefully
                        continue

                    cache_delta = src_cache_seq_len - target_seq_len

                    span_inputs: dict[str, Any] = {
                        "condition": cond,
                        "sample_id": sample_id,
                        "sample_key": s_key,
                        "source_sample_id": source_sample_id,
                        "source_sample_key": source_s_key,
                        "model": args.model_name,
                        "latent_steps": args.latent_steps,
                        "cache_present": cache_present,
                        "target_cache_seq_len": target_seq_len,
                        "source_cache_seq_len": src_cache_seq_len,
                        "cache_seq_len_delta": cache_delta,
                    }
                    if cond == InterventionCondition.CROSS and cross_j is not None:
                        span_inputs["cross_source_question"] = dataset_iter[
                            cross_j
                        ].get("question", "")

                    cond_span_ctx = (
                        self.tracker_port.start_span(
                            name=f"decode_{cond}",
                            span_type="LLM",
                            inputs=span_inputs,
                        )
                        if self.tracker_port
                        else contextlib.nullcontext(None)
                    )

                    with cond_span_ctx as c_span:
                        t0 = time.time()
                        decode_res = method.decode_with_context(
                            [item],
                            past_kv=chosen_context,
                            initial_traces=chosen_traces,
                        )
                        latency = time.time() - t0
                        condition_latencies[cond].append(latency)

                        res = decode_res[0]
                        pred = res.get("prediction")
                        raw_pred = res.get("raw_prediction", "")
                        is_correct = bool(res.get("correct", False))
                        err_msg = res.get("error_msg") or None
                        cond_preds[cond] = (pred, is_correct, raw_pred)

                        prompt_tokens = len(
                            res.get("agents", [{}])[-1].get("input_ids", [])
                        )
                        generated_tokens = (
                            len(
                                model.tokenizer.encode(
                                    raw_pred, add_special_tokens=False
                                )
                            )
                            if hasattr(model, "tokenizer")
                            else len(raw_pred.split())
                        )
                        total_tokens = prompt_tokens + generated_tokens
                        condition_tokens[cond].append(generated_tokens)

                        if c_span is not None:
                            c_span.set_token_usage(
                                prompt_tokens=prompt_tokens,
                                completion_tokens=generated_tokens,
                            )
                            c_span.set_outputs(
                                {
                                    "condition": cond,
                                    "sample_key": s_key,
                                    "source_sample_key": source_s_key,
                                    "prediction": pred,
                                    "correct": is_correct,
                                    "generated_tokens": generated_tokens,
                                    "prompt_tokens": prompt_tokens,
                                    "total_tokens": total_tokens,
                                    "latency_sec": round(latency, 4),
                                    "cache_seq_len": src_cache_seq_len,
                                    "raw_prediction": raw_pred,
                                }
                            )
                            if err_msg:
                                c_span.set_status("ERROR", description=err_msg)

                        record = SampleInterventionRecord(
                            sample_id=sample_id,
                            source_sample_id=source_sample_id,
                            condition=cond,
                            question=question,
                            gold=gold,
                            prediction=pred,
                            raw_prediction=raw_pred,
                            correct=is_correct,
                            model=args.model_name,
                            task=args.task,
                            latent_steps=args.latent_steps,
                            seed=sample_seed,
                            sample_index=sample_index,
                            sample_key=s_key,
                            source_sample_index=source_sample_index,
                            source_sample_key=source_s_key,
                            target_cache_seq_len=target_seq_len,
                            source_cache_seq_len=src_cache_seq_len,
                            cache_seq_len_delta=cache_delta,
                            cache_present=cache_present,
                            num_layers=_num_layers,
                            cache_dtype=_cache_dtype,
                            latency=round(latency, 4),
                            generated_tokens=generated_tokens,
                            error=err_msg,
                        )
                        sample_records.append(record)
                        results_by_sample_and_cond[i][cond] = record

                    if cond != InterventionCondition.OWN:
                        del chosen_context
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del own_context_gpu
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Root trace outputs
                root_outputs: dict[str, Any] = {
                    "sample_index": sample_index,
                    "sample_key": s_key,
                }
                for cond in conditions:
                    if cond in cond_preds:
                        p, c_ok, _ = cond_preds[cond]
                        root_outputs[f"{cond}_prediction"] = p
                        root_outputs[f"{cond}_correct"] = c_ok
                if "cross" in cond_preds:
                    root_outputs["cross_source_sample_id"] = cross_source_sample_id
                    root_outputs["cross_source_sample_key"] = cross_source_key

                if root_span is not None:
                    root_span.set_outputs(root_outputs)
                    resp_parts = [
                        f"{cond}: {cond_preds[cond][0]} "
                        f"({'PASS' if cond_preds[cond][1] else 'FAIL'})"
                        for cond in conditions
                        if cond in cond_preds
                    ]
                    resp_preview = " | ".join(resp_parts)
                    if self.tracker_port:
                        self.tracker_port.update_current_trace(
                            response_preview=resp_preview
                        )

                trace_id = (
                    getattr(root_span, "trace_id", None)
                    if root_span is not None
                    else None
                )

            # GenAI Assessments
            if self.tracker_port and trace_id:
                self.tracker_port.log_expectation(
                    trace_id=trace_id,
                    name="expected_answer",
                    value=gold,
                    source_id="ground_truth",
                )
                solution_text = item.get("solution")
                if solution_text:
                    self.tracker_port.log_expectation(
                        trace_id=trace_id,
                        name="reference_solution",
                        value=str(solution_text),
                        source_id="ground_truth",
                    )
                sample_has_error = any(
                    rec.error is not None
                    for rec in results_by_sample_and_cond[i].values()
                )
                self.tracker_port.log_feedback(
                    trace_id=trace_id,
                    name="execution_success",
                    value=not sample_has_error,
                    source_id="default",
                    rationale="sample status=success"
                    if not sample_has_error
                    else "sample status=error",
                )
                for cond in conditions:
                    if cond in cond_preds:
                        _, c_ok, _ = cond_preds[cond]
                        self.tracker_port.log_feedback(
                            trace_id=trace_id,
                            name=f"{cond}_correct",
                            value=c_ok,
                            source_id="evaluator",
                        )
                if "own" in cond_preds:
                    own_p, own_c, _ = cond_preds["own"]
                    self.tracker_port.log_feedback(
                        trace_id=trace_id,
                        name="runtime_correctness",
                        value=own_c,
                        source_id="default",
                        rationale=f"normalized prediction='{own_p}'; expected='{gold}'",
                    )

        # ── Phase 3: Metric aggregation ────────────────────────────────────────
        total_runtime = time.time() - start_total_time
        runtime_per_sample = total_runtime / n_samples if n_samples > 0 else 0.0

        def _n_correct(cond: str) -> int:
            return sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get(cond)
                and results_by_sample_and_cond[ii][cond].correct
            )

        def _change_rate(cond: str) -> float:
            changed = sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get(cond)
                and results_by_sample_and_cond[ii].get("own")
                and results_by_sample_and_cond[ii][cond].prediction
                != results_by_sample_and_cond[ii]["own"].prediction
            )
            return changed / n_samples if n_samples > 0 else 0.0

        # Per-condition accuracy
        n_correct_by_cond: dict[str, int] = {
            cond: _n_correct(cond) for cond in conditions
        }
        acc_by_cond: dict[str, float] = {
            cond: n_correct_by_cond[cond] / n_samples if n_samples > 0 else 0.0
            for cond in conditions
        }

        acc_own = acc_by_cond.get("own", 0.0)
        n_own_correct = n_correct_by_cond.get("own", 0)

        # Change rates
        change_rate_by_cond: dict[str, float] = {
            cond: _change_rate(cond) for cond in conditions if cond != "own"
        }

        # Paired transitions & harm/rescue per intervention condition
        paired_transitions: dict[str, Any] = {}
        harm_rescue: dict[str, dict[str, int]] = {}
        mcnemar_dict: dict[str, Any] = {}
        correctness_patterns: dict[str, int] = {}

        for cond in conditions:
            if cond == "own":
                continue
            # All 4 cells
            own_c_cond_w = sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get("own")
                and results_by_sample_and_cond[ii].get(cond)
                and results_by_sample_and_cond[ii]["own"].correct
                and not results_by_sample_and_cond[ii][cond].correct
            )
            own_w_cond_c = sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get("own")
                and results_by_sample_and_cond[ii].get(cond)
                and not results_by_sample_and_cond[ii]["own"].correct
                and results_by_sample_and_cond[ii][cond].correct
            )
            own_c_cond_c = sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get("own")
                and results_by_sample_and_cond[ii].get(cond)
                and results_by_sample_and_cond[ii]["own"].correct
                and results_by_sample_and_cond[ii][cond].correct
            )
            own_w_cond_w = sum(
                1
                for ii in range(n_samples)
                if results_by_sample_and_cond[ii].get("own")
                and results_by_sample_and_cond[ii].get(cond)
                and not results_by_sample_and_cond[ii]["own"].correct
                and not results_by_sample_and_cond[ii][cond].correct
            )
            paired_transitions[f"own_vs_{cond}"] = {
                "own_correct_cond_correct": own_c_cond_c,
                "own_correct_cond_wrong": own_c_cond_w,
                "own_wrong_cond_correct": own_w_cond_c,
                "own_wrong_cond_wrong": own_w_cond_w,
            }
            harm_rescue[cond] = {
                "harm": own_c_cond_w,
                "rescue": own_w_cond_c,
            }
            mcnemar_dict[f"own_vs_{cond}"] = compute_mcnemar_test(
                b=own_c_cond_w, c=own_w_cond_c
            )

        # Correctness patterns across all conditions per sample
        for ii in range(n_samples):
            pattern_parts = []
            for cond in conditions:
                rec = results_by_sample_and_cond[ii].get(cond)
                val = 1 if (rec and rec.correct) else 0
                pattern_parts.append(f"{cond}={val}")
            pattern = " ".join(pattern_parts)
            correctness_patterns[pattern] = correctness_patterns.get(pattern, 0) + 1

        # Condition runtimes and token stats
        runtime_by_cond: dict[str, float] = {
            cond: sum(condition_latencies.get(cond, [])) for cond in conditions
        }
        tokens_mean: dict[str, float] = {}
        tokens_max: dict[str, int] = {}
        truncation_rates: dict[str, float] = {}
        max_tok = int(getattr(args, "max_new_tokens", 0))
        for cond in conditions:
            toks = condition_tokens.get(cond, [])
            if toks:
                tokens_mean[cond] = round(sum(toks) / len(toks), 2)
                tokens_max[cond] = max(toks)
                truncation_rates[cond] = (
                    round(sum(1 for t in toks if t >= max_tok) / len(toks), 4)
                    if max_tok > 0
                    else 0.0
                )

        # RAM usage stats
        if cache_bytes_list:
            avg_bytes = sum(cache_bytes_list) / len(cache_bytes_list)
            max_bytes = max(cache_bytes_list)
            ram_stats: dict[str, Any] = {
                "avg_cache_bytes_per_sample": int(avg_bytes),
                "max_cache_bytes_per_sample": max_bytes,
                "estimated_ram_100_samples_mb": round(avg_bytes * 100 / (1024**2), 2),
                "estimated_ram_300_samples_mb": round(avg_bytes * 300 / (1024**2), 2),
                "n_samples_measured": n_samples,
            }
        else:
            ram_stats = {}

        # Cache sequence length stats
        if context_seq_lens:
            csl = context_seq_lens
            cache_seq_stats: dict[str, Any] = {
                "own_seq_len_mean": round(sum(csl) / len(csl), 2),
                "own_seq_len_min": min(csl),
                "own_seq_len_max": max(csl),
            }
            if "cross" in conditions and cross_indices:
                cross_src_lens = [
                    context_seq_lens[cross_indices[ii]] for ii in range(n_samples)
                ]
                deltas = [abs(csl[ii] - cross_src_lens[ii]) for ii in range(n_samples)]
                cache_seq_stats.update(
                    {
                        "cross_src_seq_len_mean": round(
                            sum(cross_src_lens) / len(cross_src_lens), 2
                        ),
                        "cross_seq_len_delta_mean": round(sum(deltas) / len(deltas), 2),
                        "cross_seq_len_delta_max": max(deltas),
                    }
                )
        else:
            cache_seq_stats = {}

        metrics = InterventionMetrics(
            accuracy_own=round(acc_by_cond.get("own", 0.0), 4),
            accuracy_cross=round(acc_by_cond.get("cross", 0.0), 4),
            accuracy_drop=round(acc_by_cond.get("drop", 0.0), 4),
            accuracy_zero=round(acc_by_cond.get("zero", 0.0), 4),
            accuracy_by_condition={c: round(v, 4) for c, v in acc_by_cond.items()},
            n_correct_by_condition=n_correct_by_cond,
            answer_change_rate_cross=round(change_rate_by_cond.get("cross", 0.0), 4),
            answer_change_rate_drop=round(change_rate_by_cond.get("drop", 0.0), 4),
            answer_change_rate_zero=round(change_rate_by_cond.get("zero", 0.0), 4),
            answer_change_rate_by_condition={
                c: round(v, 4) for c, v in change_rate_by_cond.items()
            },
            accuracy_delta_own_cross=round(acc_own - acc_by_cond.get("cross", 0.0), 4),
            accuracy_delta_own_drop=round(acc_own - acc_by_cond.get("drop", 0.0), 4),
            accuracy_delta_own_zero=round(acc_own - acc_by_cond.get("zero", 0.0), 4),
            n_samples=n_samples,
            n_own_correct=n_own_correct,
            n_cross_correct=n_correct_by_cond.get("cross", 0),
            n_drop_correct=n_correct_by_cond.get("drop", 0),
            n_zero_correct=n_correct_by_cond.get("zero", 0),
            runtime_total=round(total_runtime, 4),
            runtime_per_sample=round(runtime_per_sample, 4),
            runtime_own=round(runtime_by_cond.get("own", 0.0), 4),
            runtime_cross=round(runtime_by_cond.get("cross", 0.0), 4),
            runtime_drop=round(runtime_by_cond.get("drop", 0.0), 4),
            runtime_zero=round(runtime_by_cond.get("zero", 0.0), 4),
            runtime_by_condition={c: round(v, 4) for c, v in runtime_by_cond.items()},
            paired_transitions=paired_transitions,
            harm_rescue=harm_rescue,
            correctness_patterns=correctness_patterns,
            mcnemar_tests=mcnemar_dict,
            cache_seq_len_stats=cache_seq_stats,
            ram_usage_stats=ram_stats,
            max_new_tokens=max_tok,
            tokens_generated_mean=tokens_mean,
            tokens_generated_max=tokens_max,
            truncation_rate=truncation_rates,
        )

        # ── Phase 4: Artifact generation & persistence ─────────────────────────
        sanitized_model = args.model_name.replace("/", "_")
        run_stem = f"intervention_{args.task}_{sanitized_model}_s{n_samples}_{int(time.time())}"

        local_cache_file = f"{run_stem}.json"
        with contextlib.suppress(Exception):
            self.cache_port.save_json(
                CacheLayer.EVALUATION_INTERVENTIONS,
                local_cache_file,
                {
                    "metrics": metrics.to_dict(),
                    "records": [r.to_dict() for r in sample_records],
                },
            )

        runtime_dir = self.cache_port.get_layer_path(CacheLayer.RUNTIME)
        sample_results_path = runtime_dir / f"{run_stem}_sample_results.jsonl"
        summary_path = runtime_dir / f"{run_stem}_summary.json"
        resolved_config_path = runtime_dir / f"{run_stem}_resolved_config.yaml"
        patterns_path = runtime_dir / f"{run_stem}_correctness_patterns.json"

        with open(sample_results_path, "w", encoding="utf-8") as f:
            for r in sample_records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        clean_args = {
            k: v
            for k, v in args_dict.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics.to_dict(),
                    "paired_transitions": paired_transitions,
                    "harm_rescue": harm_rescue,
                    "correctness_patterns": correctness_patterns,
                    "mcnemar_tests": mcnemar_dict,
                    "cache_seq_len_stats": cache_seq_stats,
                    "ram_usage_stats": ram_stats,
                    "config": clean_args,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(resolved_config_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_args, f, allow_unicode=True, default_flow_style=False)

        with open(patterns_path, "w", encoding="utf-8") as f:
            json.dump(correctness_patterns, f, ensure_ascii=False, indent=2)

        # ── Phase 5: Complete MLflow tracking ──────────────────────────────────
        if self.tracker_port:
            self.tracker_port.log_metrics(metrics.to_mlflow_metrics())
            self.tracker_port.log_artifact(sample_results_path, artifact_path="results")
            self.tracker_port.log_artifact(summary_path, artifact_path="results")
            self.tracker_port.log_artifact(patterns_path, artifact_path="results")
            self.tracker_port.log_artifact(resolved_config_path, artifact_path="config")
            self.tracker_port.flush_traces()
            self.tracker_port.end_run(status="FINISHED")

        return metrics, sample_records
