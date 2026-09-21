import contextlib
import time
from typing import Any

from tqdm import tqdm

from ..domain.models import BenchmarkMetrics
from ..domain.ports.cache_port import CacheLayer, CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.model_port import ModelPort
from ..domain.services.baseline import BaselineMethod
from ..domain.services.latent_mas import LatentMASMethod
from ..domain.services.text_mas import TextMASMethod
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from .evaluation_service import evaluate_predictions


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
    ) -> None:
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR

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
            print("Question:")
            print(res.get("question", "").strip())
            agents = res.get("agents", [])
            for a in agents:
                name = a.get("name", "Agent")
                role = a.get("role", "")
                agent_header = f"----- Agent: {name} ({role}) -----"
                print(agent_header)
                agent_input = a.get("input", "").rstrip()
                agent_output = a.get("output", "").rstrip()
                latent_steps = a.get("latent_steps", None)
                print("[To Tokenize]")
                print(agent_input)
                if latent_steps is not None:
                    print("[Latent Steps]")
                    print(latent_steps)
                print("[Output]")
                print(agent_output)
                print("----------------------------------------------")
            print(
                f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}"
            )

        processed += len(results)
        if progress is not None:
            progress.update(len(results))
        return processed, preds

    def execute(self, model: ModelWrapper, args: Any) -> tuple[BenchmarkMetrics, list[dict]]:
        method = self.create_method(model, args)
        start_time = time.time()

        dataset_iter = self.dataset_port.load(task=args.task, split=args.split)

        if args.max_samples == -1:
            dataset_iter = list(dataset_iter)
            args.max_samples = len(dataset_iter)

        progress = tqdm(total=args.max_samples)
        preds: list[dict] = []
        processed = 0
        batch: list[dict] = []

        for item in dataset_iter:
            if processed >= args.max_samples:
                break
            batch.append(item)
            if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
                processed, preds = self.process_batch(
                    method,
                    batch,
                    processed,
                    preds,
                    progress,
                    args.max_samples,
                    args,
                )
                batch = []
                if processed >= args.max_samples:
                    break

        if batch and processed < args.max_samples:
            processed, preds = self.process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples=args.max_samples,
                args=args,
            )
        progress.close()

        total_time = time.time() - start_time
        acc, correct = evaluate_predictions(preds)

        metrics = BenchmarkMetrics(
            method=args.method,
            model=args.model_name,
            split=args.split,
            seed=args.seed,
            max_samples=args.max_samples,
            accuracy=acc,
            correct=correct,
            total_time_sec=round(total_time, 4),
            time_per_sample_sec=round(total_time / args.max_samples, 4) if args.max_samples > 0 else 0.0,
        )

        # Cache run results to root .cache/evaluation/runs/
        sanitized_model = args.model_name.replace("/", "_")
        run_cache_filename = (
            f"run_{args.task}_{args.method}_{sanitized_model}_s{args.max_samples}.json"
        )
        with contextlib.suppress(Exception):
            self.cache_port.save_json(
                CacheLayer.EVALUATION_RUNS,
                run_cache_filename,
                {
                    "metrics": metrics.to_dict(),
                    "predictions": preds,
                },
            )

        return metrics, preds
