import contextlib
import json
import random
import time
from typing import Any

import torch
import yaml
from tqdm import tqdm

from ..domain.models import (
    InterventionCondition,
    InterventionMetrics,
    SampleInterventionRecord,
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
    n: int, policy: str = "shift_1", seed: int = 42
) -> list[int]:
    """Generate deterministic permutation indices for cross-sample pairing.

    Guarantees that index[i] != i for all i (derangement) when n >= 2.
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
        # Find a valid derangement deterministically
        for _ in range(1000):
            perm = list(indices)
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm
        # Fallback to shift_1 if derangement search takes too long
        return [(i + 1) % n for i in range(n)]

    # Default to shift_1
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


class InterventionUseCase:
    """Application use case for latent communication intervention experiments.

    Evaluates three conditions:
    1. own: q_i -> H_i -> judger
    2. cross: q_i -> H_j -> judger (j != i)
    3. zero: q_i -> 0 -> judger
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
            args, "intervention_conditions", ["own", "cross", "zero"]
        )
        if isinstance(raw_conditions, str):
            conditions = [c.strip() for c in raw_conditions.split(",") if c.strip()]
        else:
            conditions = list(raw_conditions)

        cross_policy = getattr(args, "cross_policy", "shift_1")
        zero_mode = getattr(args, "zero_mode", "none")  # "none" or "zeros"
        save_raw_cache = bool(getattr(args, "save_raw_cache", False))

        # 1. Load problem samples
        dataset_iter = list(self.dataset_port.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            dataset_iter = dataset_iter[: args.max_samples]
        n_samples = len(dataset_iter)

        if "cross" in conditions and n_samples < 2:
            raise ValueError(
                f"Intervention condition 'cross' requires at least 2 samples, got {n_samples}."
            )

        cross_indices = (
            generate_cross_indices(n_samples, policy=cross_policy, seed=args.seed)
            if "cross" in conditions
            else []
        )

        # 2. Instantiate LatentMASMethod
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

        # Start tracking run early so active run ID and experiment are bound for GenAI tracing
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
                    "split": args.split,
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
                    "zero_mode": zero_mode,
                    "device": str(model.device),
                }
            )

        print(f"\n[Intervention] Starting pilot on {n_samples} samples...")
        print(
            f"Conditions: {conditions} | Cross policy: {cross_policy} | Zero mode: {zero_mode}"
        )

        # Phase 1: Build latent communication contexts H_i for each sample
        print("\n--- Phase 1: Building latent communication contexts ---")
        contexts: list[Any] = []
        traces_list: list[list[dict]] = []
        context_build_latencies: list[float] = []
        context_seq_lens: list[int] = []

        for idx, item in enumerate(tqdm(dataset_iter, desc="Building latent contexts")):
            t_start = time.time()
            past_kv, agent_traces = method.build_latent_contexts([item])
            t_build = time.time() - t_start

            # Offload past_kv to CPU to avoid GPU OOM on large models (e.g. 7B)
            past_kv_cpu = move_past_kv(past_kv, "cpu")
            contexts.append(past_kv_cpu)
            traces_list.append(agent_traces[0] if agent_traces else [])
            context_build_latencies.append(t_build)
            context_seq_lens.append(get_kv_sequence_length(past_kv_cpu))

            # Optionally cache raw KV tensors locally
            if save_raw_cache:
                cache_name = f"latent_{args.task}_{idx}_s{args.seed}.pt"
                with contextlib.suppress(Exception):
                    self.cache_port.save_torch(
                        CacheLayer.LATENT_INTERVENTIONS, cache_name, past_kv_cpu
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Phase 2: Decoding under own, cross, and zero conditions with GenAI tracing
        print("\n--- Phase 2: Decoding under intervention conditions ---")
        sample_records: list[SampleInterventionRecord] = []
        results_by_sample_and_cond: dict[int, dict[str, SampleInterventionRecord]] = {
            i: {} for i in range(n_samples)
        }
        condition_latencies: dict[str, list[float]] = {c: [] for c in conditions}
        condition_tokens: dict[str, list[int]] = {c: [] for c in conditions}

        for i, item in enumerate(tqdm(dataset_iter, desc="Intervention decoding")):
            sample_id = f"sample_{i}"
            gold = str(item.get("gold", ""))
            question = item.get("question", "")

            cross_j = cross_indices[i] if "cross" in conditions else None
            cross_source_sample_id = (
                f"sample_{cross_j}" if cross_j is not None else None
            )

            # Dictionary to collect results for root trace outputs
            cond_preds: dict[str, tuple[str | None, bool, str]] = {}

            # Trace root span context
            trace_ctx = (
                self.tracker_port.start_sample_trace(
                    name="latentmas_sample",
                    inputs={
                        "sample_id": sample_id,
                        "question": question,
                        "gold": gold,
                    },
                    tags={
                        "task": args.task,
                        "model": args.model_name,
                        "method": args.method,
                        "sample_id": sample_id,
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
                # 1. Child span: build_own_context
                build_span_ctx = (
                    self.tracker_port.start_span(
                        name="build_own_context",
                        span_type="CHAIN",
                        inputs={
                            "sample_id": sample_id,
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
                                "cache_source_sample_id": sample_id,
                                "cache_sequence_length": context_seq_lens[i],
                                "cache_format": "DynamicCache"
                                if hasattr(contexts[i], "get_seq_length")
                                else "past_kv",
                                "build_latency_sec": round(
                                    context_build_latencies[i], 4
                                ),
                                "agent_prompts": agent_prompts,
                            }
                        )

                # 2. Condition decodings
                for cond in conditions:
                    sample_seed = args.seed + i
                    random.seed(sample_seed)
                    torch.manual_seed(sample_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sample_seed)

                    source_sample_id: str | int | None = None
                    chosen_context: Any = None
                    chosen_traces: list[list[dict]] | None = None
                    cache_present = False
                    cache_seq_len = 0
                    cache_fmt = "none"

                    if cond == InterventionCondition.OWN:
                        source_sample_id = sample_id
                        chosen_context = own_context_gpu
                        chosen_traces = own_traces
                        cache_present = True
                        cache_seq_len = context_seq_lens[i]
                        cache_fmt = "past_kv"

                    elif cond == InterventionCondition.CROSS:
                        source_sample_id = cross_source_sample_id
                        ctx_cross_cloned = clone_past_kv(contexts[cross_j])
                        chosen_context = move_past_kv(ctx_cross_cloned, model.device)
                        chosen_traces = None
                        cache_present = True
                        cache_seq_len = context_seq_lens[cross_j]
                        cache_fmt = "past_kv"

                    elif cond == InterventionCondition.ZERO:
                        source_sample_id = None
                        if zero_mode == "zeros":
                            ctx_zero = create_zero_past_kv(contexts[i])
                            chosen_context = move_past_kv(ctx_zero, model.device)
                            cache_present = True
                            cache_seq_len = context_seq_lens[i]
                            cache_fmt = "zeros"
                        else:
                            chosen_context = None
                            cache_present = False
                            cache_seq_len = 0
                            cache_fmt = "none"
                        chosen_traces = None

                    span_inputs = {
                        "condition": cond,
                        "sample_id": sample_id,
                        "source_sample_id": source_sample_id,
                        "model": args.model_name,
                        "latent_steps": args.latent_steps,
                        "cache_present": cache_present,
                        "cache_source_sample_id": source_sample_id,
                        "cache_sequence_length": cache_seq_len,
                        "cache_format": cache_fmt,
                    }
                    if cond == InterventionCondition.CROSS and cross_j is not None:
                        span_inputs["cross_source_question"] = dataset_iter[cross_j].get(
                            "question", ""
                        )

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
                                    "prediction": pred,
                                    "correct": is_correct,
                                    "generated_tokens": generated_tokens,
                                    "prompt_tokens": prompt_tokens,
                                    "total_tokens": total_tokens,
                                    "latency_sec": round(latency, 4),
                                    "raw_prediction": raw_pred,
                                }
                            )
                            if res.get("error_msg"):
                                c_span.set_status("ERROR", description=res["error_msg"])

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
                            latency=round(latency, 4),
                            generated_tokens=generated_tokens,
                            error=res.get("error_msg"),
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

                own_p, own_c, _ = cond_preds.get("own", (None, False, ""))
                cross_p, cross_c, _ = cond_preds.get("cross", (None, False, ""))
                zero_p, zero_c, _ = cond_preds.get("zero", (None, False, ""))

                if root_span is not None:
                    root_span.set_outputs(
                        {
                            "own_prediction": own_p,
                            "own_correct": own_c,
                            "cross_prediction": cross_p,
                            "cross_correct": cross_c,
                            "cross_source_sample_id": cross_source_sample_id,
                            "zero_prediction": zero_p,
                            "zero_correct": zero_c,
                        }
                    )
                    resp_preview = (
                        f"own: {own_p} ({'PASS' if own_c else 'FAIL'}) | "
                        f"cross: {cross_p} ({'PASS' if cross_c else 'FAIL'}) | "
                        f"zero: {zero_p} ({'PASS' if zero_c else 'FAIL'})"
                    )
                    if self.tracker_port:
                        self.tracker_port.update_current_trace(
                            response_preview=resp_preview
                        )

                trace_id = (
                    getattr(root_span, "trace_id", None)
                    if root_span is not None
                    else None
                )

            # GenAI Assessments on the completed trace
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
                if "own" in cond_preds:
                    own_p, own_c, _ = cond_preds["own"]
                    self.tracker_port.log_feedback(
                        trace_id=trace_id,
                        name="runtime_correctness",
                        value=own_c,
                        source_id="default",
                        rationale=f"normalized prediction='{own_p}'; expected='{gold}'",
                    )
                    self.tracker_port.log_feedback(
                        trace_id=trace_id,
                        name="own_correct",
                        value=own_c,
                        source_id="evaluator",
                    )
                if "cross" in cond_preds:
                    self.tracker_port.log_feedback(
                        trace_id=trace_id,
                        name="cross_correct",
                        value=cond_preds["cross"][1],
                        source_id="evaluator",
                    )
                if "zero" in cond_preds:
                    self.tracker_port.log_feedback(
                        trace_id=trace_id,
                        name="zero_correct",
                        value=cond_preds["zero"][1],
                        source_id="evaluator",
                    )

        total_runtime = time.time() - start_total_time
        runtime_per_sample = total_runtime / n_samples if n_samples > 0 else 0.0

        # Phase 3: Metric aggregation
        n_own_correct = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("own")
            and results_by_sample_and_cond[i]["own"].correct
        )
        n_cross_correct = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("cross")
            and results_by_sample_and_cond[i]["cross"].correct
        )
        n_zero_correct = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("zero")
            and results_by_sample_and_cond[i]["zero"].correct
        )

        acc_own = n_own_correct / n_samples if n_samples > 0 else 0.0
        acc_cross = n_cross_correct / n_samples if n_samples > 0 else 0.0
        acc_zero = n_zero_correct / n_samples if n_samples > 0 else 0.0

        # Answer change rate compared to own prediction
        changed_cross_count = sum(
            1
            for i in range(n_samples)
            if (
                results_by_sample_and_cond[i].get("cross")
                and results_by_sample_and_cond[i].get("own")
                and (
                    results_by_sample_and_cond[i]["cross"].prediction
                    != results_by_sample_and_cond[i]["own"].prediction
                )
            )
        )
        changed_zero_count = sum(
            1
            for i in range(n_samples)
            if (
                results_by_sample_and_cond[i].get("zero")
                and results_by_sample_and_cond[i].get("own")
                and (
                    results_by_sample_and_cond[i]["zero"].prediction
                    != results_by_sample_and_cond[i]["own"].prediction
                )
            )
        )

        change_rate_cross = changed_cross_count / n_samples if n_samples > 0 else 0.0
        change_rate_zero = changed_zero_count / n_samples if n_samples > 0 else 0.0

        # Paired transition counts
        own_c_cross_w = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("own")
            and results_by_sample_and_cond[i].get("cross")
            and results_by_sample_and_cond[i]["own"].correct
            and not results_by_sample_and_cond[i]["cross"].correct
        )
        own_w_cross_c = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("own")
            and results_by_sample_and_cond[i].get("cross")
            and not results_by_sample_and_cond[i]["own"].correct
            and results_by_sample_and_cond[i]["cross"].correct
        )
        own_c_zero_w = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("own")
            and results_by_sample_and_cond[i].get("zero")
            and results_by_sample_and_cond[i]["own"].correct
            and not results_by_sample_and_cond[i]["zero"].correct
        )
        own_w_zero_c = sum(
            1
            for i in range(n_samples)
            if results_by_sample_and_cond[i].get("own")
            and results_by_sample_and_cond[i].get("zero")
            and not results_by_sample_and_cond[i]["own"].correct
            and results_by_sample_and_cond[i]["zero"].correct
        )

        paired_transitions = {
            "own_correct_to_cross_wrong": own_c_cross_w,
            "own_wrong_to_cross_correct": own_w_cross_c,
            "own_correct_to_zero_wrong": own_c_zero_w,
            "own_wrong_to_zero_correct": own_w_zero_c,
        }

        # McNemar tests
        mcnemar_own_cross = compute_mcnemar_test(b=own_c_cross_w, c=own_w_cross_c)
        mcnemar_own_zero = compute_mcnemar_test(b=own_c_zero_w, c=own_w_zero_c)

        mcnemar_dict = {
            "own_vs_cross": mcnemar_own_cross,
            "own_vs_zero": mcnemar_own_zero,
        }

        # Condition runtimes
        runtime_own = sum(condition_latencies.get("own", []))
        runtime_cross = sum(condition_latencies.get("cross", []))
        runtime_zero = sum(condition_latencies.get("zero", []))

        # Token metrics
        tokens_mean: dict[str, float] = {}
        tokens_max: dict[str, int] = {}
        truncation_rates: dict[str, float] = {}
        max_tok = int(getattr(args, "max_new_tokens", 0))
        for c in conditions:
            toks = condition_tokens.get(c, [])
            if toks:
                tokens_mean[c] = round(sum(toks) / len(toks), 2)
                tokens_max[c] = max(toks)
                truncation_rates[c] = (
                    round(sum(1 for t in toks if t >= max_tok) / len(toks), 4)
                    if max_tok > 0
                    else 0.0
                )

        metrics = InterventionMetrics(
            accuracy_own=round(acc_own, 4),
            accuracy_cross=round(acc_cross, 4),
            accuracy_zero=round(acc_zero, 4),
            answer_change_rate_cross=round(change_rate_cross, 4),
            answer_change_rate_zero=round(change_rate_zero, 4),
            accuracy_delta_own_cross=round(acc_own - acc_cross, 4),
            accuracy_delta_own_zero=round(acc_own - acc_zero, 4),
            n_samples=n_samples,
            n_own_correct=n_own_correct,
            n_cross_correct=n_cross_correct,
            n_zero_correct=n_zero_correct,
            runtime_total=round(total_runtime, 4),
            runtime_per_sample=round(runtime_per_sample, 4),
            runtime_own=round(runtime_own, 4),
            runtime_cross=round(runtime_cross, 4),
            runtime_zero=round(runtime_zero, 4),
            paired_transitions=paired_transitions,
            mcnemar_tests=mcnemar_dict,
            max_new_tokens=max_tok,
            tokens_generated_mean=tokens_mean,
            tokens_generated_max=tokens_max,
            truncation_rate=truncation_rates,
        )

        # Phase 4: Artifact generation & persistence
        sanitized_model = args.model_name.replace("/", "_")
        run_stem = f"intervention_{args.task}_{sanitized_model}_s{n_samples}_{int(time.time())}"

        # Save local execution cache
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

        # Write artifacts for MLflow
        runtime_dir = self.cache_port.get_layer_path(CacheLayer.RUNTIME)
        sample_results_path = runtime_dir / f"{run_stem}_sample_results.jsonl"
        summary_path = runtime_dir / f"{run_stem}_summary.json"
        resolved_config_path = runtime_dir / f"{run_stem}_resolved_config.yaml"

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
                    "mcnemar_tests": mcnemar_dict,
                    "config": clean_args,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(resolved_config_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_args, f, allow_unicode=True, default_flow_style=False)

        # Phase 5: Complete MLflow tracking and flush traces
        if self.tracker_port:
            self.tracker_port.log_metrics(metrics.to_mlflow_metrics())
            self.tracker_port.log_artifact(sample_results_path, artifact_path="results")
            self.tracker_port.log_artifact(summary_path, artifact_path="results")
            self.tracker_port.log_artifact(resolved_config_path, artifact_path="config")
            self.tracker_port.flush_traces()
            self.tracker_port.end_run()

        return metrics, sample_records
