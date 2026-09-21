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
        raw_conditions = getattr(args, "intervention_conditions", ["own", "cross", "zero"])
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

        print(f"\n[Intervention] Starting pilot on {n_samples} samples...")
        print(f"Conditions: {conditions} | Cross policy: {cross_policy} | Zero mode: {zero_mode}")

        # Phase 1: Build latent communication contexts H_i for each sample
        print("\n--- Phase 1: Building latent communication contexts ---")
        contexts: list[Any] = []
        traces_list: list[list[dict]] = []

        for idx, item in enumerate(tqdm(dataset_iter, desc="Building latent contexts")):
            # Build latent context
            past_kv, agent_traces = method.build_latent_contexts([item])

            # Offload past_kv to CPU to avoid GPU OOM on large models (e.g. 7B)
            past_kv_cpu = move_past_kv(past_kv, "cpu")
            contexts.append(past_kv_cpu)
            traces_list.append(agent_traces[0] if agent_traces else [])

            # Optionally cache raw KV tensors locally
            if save_raw_cache:
                cache_name = f"latent_{args.task}_{idx}_s{args.seed}.pt"
                with contextlib.suppress(Exception):
                    self.cache_port.save_torch(
                        CacheLayer.LATENT_INTERVENTIONS, cache_name, past_kv_cpu
                    )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Phase 2: Decoding under own, cross, and zero conditions
        print("\n--- Phase 2: Decoding under intervention conditions ---")
        sample_records: list[SampleInterventionRecord] = []
        results_by_sample_and_cond: dict[int, dict[str, SampleInterventionRecord]] = {
            i: {} for i in range(n_samples)
        }

        for i, item in enumerate(tqdm(dataset_iter, desc="Intervention decoding")):
            sample_id = f"sample_{i}"
            gold = item.get("gold", "")
            question = item.get("question", "")

            for cond in conditions:
                # Controlled decoding seed per sample to eliminate decoding randomness
                sample_seed = args.seed + i
                random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

                source_sample_id: str | int | None = None
                chosen_context: Any = None
                chosen_traces: list[list[dict]] | None = None

                if cond == InterventionCondition.OWN:
                    source_sample_id = sample_id
                    ctx_cloned = clone_past_kv(contexts[i])
                    chosen_context = move_past_kv(ctx_cloned, model.device)
                    chosen_traces = [traces_list[i]]

                elif cond == InterventionCondition.CROSS:
                    j = cross_indices[i]
                    source_sample_id = f"sample_{j}"
                    ctx_cloned = clone_past_kv(contexts[j])
                    chosen_context = move_past_kv(ctx_cloned, model.device)
                    chosen_traces = None

                elif cond == InterventionCondition.ZERO:
                    source_sample_id = None
                    if zero_mode == "zeros":
                        ctx_zero = create_zero_past_kv(contexts[i])
                        chosen_context = move_past_kv(ctx_zero, model.device)
                    else:
                        chosen_context = None
                    chosen_traces = None

                # Execute judger decoding with timing
                t0 = time.time()
                decode_res = method.decode_with_context(
                    [item], past_kv=chosen_context, initial_traces=chosen_traces
                )
                latency = time.time() - t0

                res = decode_res[0]
                pred = res.get("prediction")
                raw_pred = res.get("raw_prediction", "")
                is_correct = bool(res.get("correct", False))

                # Estimate generated tokens
                generated_tokens = (
                    len(model.tokenizer.encode(raw_pred, add_special_tokens=False))
                    if hasattr(model, "tokenizer")
                    else len(raw_pred.split())
                )

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

                # Clean up GPU context memory
                del chosen_context
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
        # Filter non-serializable objects from args
        clean_args = {
            k: v
            for k, v in args_dict.items()
            if isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metrics": metrics.to_dict(),
                    "config": clean_args,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        with open(resolved_config_path, "w", encoding="utf-8") as f:
            yaml.dump(clean_args, f, allow_unicode=True, default_flow_style=False)

        # Phase 5: MLflow tracking
        if self.tracker_port:
            exp_name = getattr(args, "tracking_experiment_name", "latentmas_intervention")
            run_name = f"{args.model_name}_{args.task}_intervention"
            self.tracker_port.start_run(
                experiment_name=exp_name,
                run_name=run_name,
                tags={
                    "model": args.model_name,
                    "task": args.task,
                    "method": args.method,
                    "experiment_type": "intervention",
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
                    "intervention_conditions": str(conditions),
                    "cross_policy": cross_policy,
                    "zero_mode": zero_mode,
                    "device": str(model.device),
                    "dtype": str(getattr(model.model, "dtype", "unknown"))
                    if hasattr(model, "model")
                    else "unknown",
                }
            )
            self.tracker_port.log_metrics(metrics.to_mlflow_metrics())
            self.tracker_port.log_artifact(sample_results_path, artifact_path="results")
            self.tracker_port.log_artifact(summary_path, artifact_path="results")
            self.tracker_port.log_artifact(resolved_config_path, artifact_path="config")
            self.tracker_port.end_run()

        return metrics, sample_records
