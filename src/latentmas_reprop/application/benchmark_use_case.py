import time
from typing import Any

import torch
from tqdm import tqdm

from ..domain.models import BenchmarkMetrics
from ..domain.ports.cache_port import CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.model_port import ModelPort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..domain.services.baseline import BaselineMethod
from ..domain.services.latent_mas import LatentMASMethod
from ..domain.services.text_mas import TextMASMethod
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.paths import (
    build_experiment_relpath,
    get_path_resolver,
    resolve_experiment_dtype_name,
)
from ..infrastructure.storage import is_condition_complete, verify_run_config
from .benchmark import (
    BenchmarkRunStore,
    build_metrics,
    config_snapshot,
    predictions_in_dataset_order,
    runtime_metadata,
    save_benchmark_artifacts,
    start_benchmark_tracker,
    trace_benchmark_sample,
)


class BenchmarkUseCase:
    """Application use case for executing benchmark experiments.

    Coordinates data loading, model inference, multi-agent evaluation,
    metrics generation, and execution caching.
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

    def create_method(self, model: ModelPort, args: Any):
        common_kwargs = {
            "temperature": args.temperature,
            "top_p": args.top_p,
        }
        if args.method == "baseline":
            return BaselineMethod(
                model,
                max_new_tokens=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                use_vllm=args.use_vllm,
                args=args,
            )
        elif args.method == "text_mas":
            return TextMASMethod(
                model,
                max_new_tokens_each=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
        elif args.method == "latent_mas":
            return LatentMASMethod(
                model,
                latent_steps=args.latent_steps,
                judger_max_new_tokens=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
        else:
            raise ValueError(f"Unknown method: {args.method}")

    def process_batch(
        self,
        method: Any,
        batch: list[dict],
        processed: int,
        preds: list[dict],
        progress: tqdm | None,
        max_samples: int,
        args: Any,
    ) -> tuple[int, list[dict]]:
        remaining = max_samples - processed
        if remaining <= 0:
            return processed, preds

        current_batch = batch[:remaining]
        batch_start_time = time.perf_counter()
        if args.method == "latent_mas" and getattr(args, "use_vllm", False):
            results = method.run_batch_vllm(current_batch)
        else:
            results = method.run_batch(current_batch)
        batch_elapsed = time.perf_counter() - batch_start_time

        if len(results) > remaining:
            results = results[:remaining]

        per_sample_latency = batch_elapsed / max(1, len(results))

        batch_start = processed
        for offset, res in enumerate(results):
            res["latency_sec"] = round(per_sample_latency, 4)
            preds.append(res)
            problem_idx = batch_start + offset + 1
            trace_benchmark_sample(self.tracker_port, args, problem_idx, res)

        processed += len(results)
        if progress is not None:
            progress.update(len(results))
        return processed, preds

    def execute(
        self, model: ModelWrapper, args: Any
    ) -> tuple[BenchmarkMetrics, list[dict]]:
        """Evaluate one resolved configuration, resuming samples already stored."""
        exp_name = getattr(args, "tracking_experiment_name", None)
        if not exp_name:
            raise ValueError(
                "MLflow tracking experiment name must be specified (--tracking_experiment_name or via config) before running benchmark."
            )

        dtype_name = str(
            getattr(model, "dtype_name", "")
        ) or resolve_experiment_dtype_name(args)
        backend = "vllm" if getattr(args, "use_vllm", False) else "transformers"
        requested_max_samples = int(args.max_samples)
        exp_relpath = build_experiment_relpath(
            args, dtype_name=dtype_name, backend=backend
        )
        args.exp_relpath = exp_relpath
        args.run_id = exp_relpath

        cand = getattr(self.cache_port, "_resolver", None) or getattr(
            self.cache_port, "resolver", None
        )
        if (
            cand
            and not callable(getattr(cand, "assert_called", None))
            and hasattr(cand, "get_experiment_dir")
        ):
            resolver = cand
        else:
            resolver = get_path_resolver()
        exp_dir = resolver.get_experiment_dir(exp_relpath)
        saved_cfg = config_snapshot(args)
        saved_cfg["requested_max_samples"] = requested_max_samples
        saved_cfg["dtype"] = dtype_name
        saved_cfg["backend"] = backend
        verify_run_config(exp_dir, saved_cfg)

        store = BenchmarkRunStore(self.cache_port, exp_relpath, path_resolver=resolver)
        start_benchmark_tracker(self.tracker_port, args)

        dataset = list(self.dataset_port.load(task=args.task, split=args.split))
        target = len(dataset) if requested_max_samples == -1 else requested_max_samples
        target = min(target, len(dataset))
        selected = dataset[:target]
        for idx, item in enumerate(selected):
            item["sample_index"] = idx

        overwrite = bool(getattr(args, "overwrite", False))
        if not overwrite:
            pending = [
                item
                for item in selected
                if not is_condition_complete(
                    resolver.get_condition_dir(
                        exp_dir, item["sample_index"], "benchmark"
                    )
                )
            ]
        else:
            pending = list(selected)

        if not pending and store.read_records():
            print(f"Reusing completed run at {exp_relpath}")
            metrics, predictions = store.load_finished(target)
            save_benchmark_artifacts(
                self.cache_port,
                self.tracker_port,
                args,
                "summary.json",
                metrics,
                predictions,
            )
            return metrics, predictions

        try:
            method = self.create_method(model, args)
            session_start = time.perf_counter()

            progress = tqdm(total=target, initial=target - len(pending))
            processed = target - len(pending)
            batch: list[dict] = []
            for item in pending:
                batch.append(item)
                filled = len(batch) == args.generate_bs
                if filled or processed + len(batch) == target:
                    processed, fresh = self.process_batch(
                        method,
                        batch,
                        processed,
                        [],
                        progress,
                        target,
                        args,
                    )
                    for offset, result in enumerate(fresh):
                        stored = dict(result)
                        sample_idx = batch[offset]["sample_index"]
                        stored["sample_index"] = sample_idx
                        stored["sample_key"] = f"{sample_idx:06d}"
                        store.append(stored, overwrite=overwrite)
                    batch = []
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            progress.close()

            eval_seconds = time.perf_counter() - session_start
            predictions = predictions_in_dataset_order(store, args, selected)
            saved_config = config_snapshot(args)
            saved_config["requested_max_samples"] = requested_max_samples
            saved_config["resolved_max_samples"] = target
            saved_config["dtype"] = dtype_name
            saved_config["backend"] = (
                "vllm" if getattr(args, "use_vllm", False) else "transformers"
            )
            metrics = build_metrics(
                args,
                predictions,
                run_id=exp_relpath,
                dtype_name=dtype_name,
                eval_seconds=eval_seconds,
                model_load_seconds=float(getattr(model, "load_time_sec", 0.0)),
            )
            store.write_complete(
                {
                    "status": "complete",
                    "run_id": exp_relpath,
                    "sample_count": len(predictions),
                    "metrics": metrics.to_dict(),
                    "config": saved_config,
                    "runtime": runtime_metadata(model, dtype_name, eval_seconds),
                }
            )
            save_benchmark_artifacts(
                self.cache_port,
                self.tracker_port,
                args,
                "summary.json",
                metrics,
                predictions,
            )
            return metrics, predictions
        except Exception:
            if self.tracker_port:
                self.tracker_port.end_run(status="FAILED")
            raise
