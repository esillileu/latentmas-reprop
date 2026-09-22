"""Latent communication intervention experiment use case.

Evaluates four conditions:
    own:  q_i + H_i                 (own latent context)
    cross:q_i + H_j (j != i)        (foreign latent context)
    drop: q_i + no past_key_values  (communication dropped)
    zero: q_i + zero-filled cache   (shape/position preserved, content zeroed)
"""

import contextlib
import sys
import time
import traceback
from typing import Any

from ..domain.models import (
    InterventionMetrics,
    SampleInterventionRecord,
    compute_sample_key,
)
from ..domain.ports.cache_port import CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..domain.services.latent_mas import LatentMASMethod
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import build_run_stem
from .intervention.algo import generate_cross_indices
from .intervention.artifacts import (
    save_intervention_artifacts,
    start_intervention_tracker,
)
from .intervention.context_builder import build_intervention_contexts
from .intervention.runner import run_intervention_decoding
from .intervention.stats import (
    calculate_intervention_metrics,
    compute_cache_and_ram_stats,
    compute_mcnemar_test,
)

__all__ = ["InterventionUseCase", "compute_mcnemar_test", "generate_cross_indices"]


class InterventionUseCase:
    """Application use case for latent communication intervention experiments."""

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
        raw_cond = getattr(
            args, "intervention_conditions", ["own", "cross", "drop", "zero"]
        )
        conditions = (
            [c.strip() for c in raw_cond.split(",") if c.strip()]
            if isinstance(raw_cond, str)
            else list(raw_cond)
        )
        cross_policy = getattr(args, "cross_policy", "shift_1")

        dataset_iter = list(self.dataset_port.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            dataset_iter = dataset_iter[: args.max_samples]
        n_samples = len(dataset_iter)

        if "cross" in conditions and n_samples < 2:
            raise ValueError(
                f"Intervention condition 'cross' requires at least 2 samples, got {n_samples}."
            )

        split_str = getattr(args, "split", "test")
        task_str = getattr(args, "task", "gsm8k")
        sample_keys = [
            compute_sample_key(task_str, split_str, item.get("question", ""))
            for item in dataset_iter
        ]

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
        start_intervention_tracker(
            self.tracker_port, args, model, n_samples, conditions, git_hash
        )

        try:
            return self._run_experiment(
                model=model,
                args=args,
                conditions=conditions,
                cross_policy=cross_policy,
                save_raw_cache=bool(getattr(args, "save_raw_cache", False)),
                dataset_iter=dataset_iter,
                n_samples=n_samples,
                sample_keys=sample_keys,
                method=method,
                start_total_time=start_total_time,
                git_hash=git_hash,
            )
        except Exception:
            exc_text = traceback.format_exc()
            print(f"[Intervention] FAILED: {exc_text}", file=sys.stderr)
            if self.tracker_port:
                with contextlib.suppress(Exception):
                    self.tracker_port.log_params({"failure_traceback": exc_text[:500]})
                with contextlib.suppress(Exception):
                    self.tracker_port.end_run(status="FAILED")
            raise

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
        method: LatentMASMethod,
        start_total_time: float,
        git_hash: str,
    ) -> tuple[InterventionMetrics, list[SampleInterventionRecord]]:
        contexts, traces, latencies, seq_lens, cache_bytes = (
            build_intervention_contexts(
                method=method,
                dataset_iter=dataset_iter,
                task=args.task,
                seed=args.seed,
                save_raw_cache=save_raw_cache,
                cache_port=self.cache_port,
            )
        )
        cross_indices = (
            generate_cross_indices(
                n_samples,
                policy=cross_policy,
                seed=args.seed,
                cache_seq_lens=seq_lens if cross_policy == "length_matched" else None,
            )
            if "cross" in conditions
            else []
        )
        records, cond_latencies, cond_tokens = run_intervention_decoding(
            model=model,
            method=method,
            args=args,
            conditions=conditions,
            cross_indices=cross_indices,
            dataset_iter=dataset_iter,
            sample_keys=sample_keys,
            contexts=contexts,
            traces_list=traces,
            context_build_latencies=latencies,
            context_seq_lens=seq_lens,
            cache_bytes_list=cache_bytes,
            tracker_port=self.tracker_port,
            git_hash=git_hash,
        )
        total_time = time.time() - start_total_time
        ram_stats, cache_stats = compute_cache_and_ram_stats(
            cache_bytes, seq_lens, cross_indices, conditions, n_samples
        )
        (
            metrics,
            paired_trans,
            harm_rescue,
            mcnemar_dict,
            patterns,
        ) = calculate_intervention_metrics(
            sample_records=records,
            conditions=conditions,
            n_samples=n_samples,
            total_runtime=total_time,
            condition_latencies=cond_latencies,
            condition_tokens=cond_tokens,
            cache_seq_stats=cache_stats,
            ram_stats=ram_stats,
            max_tok=int(getattr(args, "max_new_tokens", 0)),
        )
        save_intervention_artifacts(
            cache_port=self.cache_port,
            tracker_port=self.tracker_port,
            args=args,
            run_stem=build_run_stem(
                "intervention", args.task, args.model_name, n_samples
            ),
            metrics=metrics,
            sample_records=records,
            paired_transitions=paired_trans,
            harm_rescue=harm_rescue,
            correctness_patterns=patterns,
            mcnemar_dict=mcnemar_dict,
            cache_seq_stats=cache_stats,
            ram_stats=ram_stats,
        )
        return metrics, records
