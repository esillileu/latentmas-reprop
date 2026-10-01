"""Paired receiver decoding across upstream latent compute levels."""

import traceback
from copy import copy
from pathlib import Path
from time import perf_counter
from typing import Any

from ..domain.ports.cache_port import CacheLayer
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import (
    build_run_stem,
    clean_config_args,
    write_json,
    write_yaml,
)
from .receiver_reasoning.metrics import (
    log_summary_metrics,
    summarize_execution,
    summarize_receiver_records,
)
from .receiver_reasoning.runner import collect_sample
from .receiver_reasoning.runtime import RuntimeMeasurements


class ReceiverReasoningUseCase:
    def __init__(
        self, dataset_port=None, cache_port=None, evaluator_port=None, tracker_port=None
    ):
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR
        self.tracker_port = tracker_port

    def execute(self, model: Any, args: Any) -> tuple[dict, list[dict]]:
        if args.method != "latent_mas" or args.use_vllm:
            raise ValueError("Receiver reasoning requires transformers LatentMAS")
        args = copy(args)
        args.temperature = 0.0
        args.top_p = 1.0
        if args.handoff_positions is not None and args.handoff_positions <= 0:
            raise ValueError("Handoff positions must be positive")
        args.handoff_mode = "full" if args.handoff_positions is None else "tail"
        args.include_no_handoff = True
        steps = args.upstream_steps
        if (
            len(steps) not in (1, 2)
            or any(step <= 0 for step in steps)
            or steps != sorted(set(steps))
        ):
            raise ValueError("Expected one or two positive increasing upstream levels")
        items = list(self.dataset_port.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            items = items[: args.max_samples]
        if not items:
            raise ValueError("Receiver reasoning requires a nonempty dataset")
        config = clean_config_args(args)
        config["model_dtype"] = getattr(model, "dtype_name", None)
        config["git_commit"] = get_git_commit_hash()
        stem = build_run_stem(
            "receiver_reasoning", args.task, args.model_name, len(items)
        )
        directory = (
            Path(self.cache_port.get_layer_path(CacheLayer.EVALUATION_RUNS)) / stem
        )
        records_path = directory / "sample_results.jsonl"
        summary_path = directory / "summary.json"
        config_path = directory / "resolved_config.yaml"
        write_yaml(config_path, config)
        tracker = self.tracker_port
        if tracker:
            tags = {
                "experiment_type": "receiver_reasoning",
                "git_commit": config["git_commit"],
            }
            if getattr(args, "version_tag", None):
                tags["version_tag"] = args.version_tag
            tracker.start_run(
                experiment_name=args.tracking_experiment_name,
                run_name=stem,
                tags=tags,
            )
            tracker.log_params(config)
        records = []
        started = perf_counter()
        runtime = RuntimeMeasurements(model)
        try:
            for index, item in enumerate(items):
                collect_sample(
                    model,
                    args,
                    item,
                    index,
                    config,
                    records,
                    records_path,
                    self.evaluator_port,
                    tracker,
                    runtime,
                )
                print(
                    f"[Receiver reasoning] {index + 1}/{len(items)} samples", flush=True
                )
            summary = summarize_receiver_records(records, steps) | {
                "config": config,
                **summarize_execution(
                    records,
                    len(items),
                    perf_counter() - started,
                    model,
                    runtime.run_peak(),
                    steps,
                ),
            }
            write_json(summary_path, summary)
            if tracker:
                tracker.log_metrics(log_summary_metrics(summary))
                for path in (records_path, summary_path, config_path):
                    tracker.log_artifact(path, artifact_path="results")
                tracker.flush_traces()
                tracker.end_run(status="FINISHED")
            print(f"Artifacts: {directory}", flush=True)
            return summary, records
        except (Exception, KeyboardInterrupt) as exc:
            failure_path = directory / "failure.json"
            failure = {
                "failure_traceback": traceback.format_exc(),
                **summarize_execution(
                    records,
                    len(items),
                    perf_counter() - started,
                    model,
                    runtime.run_peak(),
                    steps,
                ),
            }
            write_json(failure_path, failure)
            if tracker:
                tracker.log_artifact(failure_path, artifact_path="results")
                tracker.log_artifact(config_path, artifact_path="results")
                tracker.log_metrics(log_summary_metrics(failure))
                if records:
                    tracker.log_artifact(records_path, artifact_path="results")
                tracker.flush_traces()
                tracker.end_run(
                    status="KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
                )
            raise
