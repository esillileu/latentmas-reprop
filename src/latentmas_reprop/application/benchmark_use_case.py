import time
from typing import Any

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
from .benchmark import (
    BenchmarkRunStore,
    build_metrics,
    build_run_id,
    config_snapshot,
    predictions_in_dataset_order,
    runtime_metadata,
    sample_key_for,
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
        if args.method == "latent_mas" and getattr(args, "use_vllm", False):
            results = method.run_batch_vllm(current_batch)
        else:
            results = method.run_batch(current_batch)

        if len(results) > remaining:
            results = results[:remaining]

        batch_start = processed
        for offset, res in enumerate(results):
            preds.append(res)
            problem_idx = batch_start + offset + 1
            print(f"\n==================== Problem #{problem_idx} ====================")
            print("Question:\n" + res.get("question", "").strip())
            for a in res.get("agents", []):
                name, role = a.get("name", "Agent"), a.get("role", "")
                print(f"----- Agent: {name} ({role}) -----")
                print("[To Tokenize]\n" + a.get("input", "").rstrip())
                if a.get("latent_steps") is not None:
                    print(f"[Latent Steps]\n{a['latent_steps']}")
                print("[Output]\n" + a.get("output", "").rstrip())
                print("----------------------------------------------")
            print(
                f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
            )
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

        dtype_name = str(getattr(model, "dtype_name", ""))
        requested_max_samples = int(args.max_samples)
        run_id = build_run_id(args, dtype_name)
        store = BenchmarkRunStore(self.cache_port, run_id)
        start_benchmark_tracker(self.tracker_port, args)

        if store.is_complete():
            print(f"Reusing completed run {run_id}")
            metrics, predictions = store.load_finished()
            save_benchmark_artifacts(
                self.cache_port,
                self.tracker_port,
                args,
                f"{run_id}.json",
                metrics,
                predictions,
            )
            return metrics, predictions

        try:
            method = self.create_method(model, args)
            dataset = list(self.dataset_port.load(task=args.task, split=args.split))
            target = (
                len(dataset) if requested_max_samples == -1 else requested_max_samples
            )
            target = min(target, len(dataset))
            selected = dataset[:target]
            done = store.completed_keys()
            pending = [
                item for item in selected if sample_key_for(args, item) not in done
            ]

            session_start = time.perf_counter()
            booked = 0.0

            def commit_eval_time() -> float:
                nonlocal booked
                elapsed = time.perf_counter() - session_start
                total = store.add_eval_time(elapsed - booked)
                booked = elapsed
                return total

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
                        stored["sample_key"] = sample_key_for(args, batch[offset])
                        store.append(stored)
                    commit_eval_time()
                    batch = []
            progress.close()
            if not pending:
                commit_eval_time()

            eval_seconds = float(store.read_progress().get("eval_time_sec", 0.0))
            predictions = predictions_in_dataset_order(store, args, selected)
            saved_config = config_snapshot(args)
            saved_config["requested_max_samples"] = requested_max_samples
            saved_config["resolved_max_samples"] = target
            metrics = build_metrics(
                args,
                predictions,
                run_id=run_id,
                dtype_name=dtype_name,
                eval_seconds=eval_seconds,
                model_load_seconds=float(getattr(model, "load_time_sec", 0.0)),
            )
            store.write_complete(
                {
                    "status": "complete",
                    "run_id": run_id,
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
                f"{run_id}.json",
                metrics,
                predictions,
            )
            return metrics, predictions
        except Exception:
            if self.tracker_port:
                self.tracker_port.end_run(status="FAILED")
            raise
