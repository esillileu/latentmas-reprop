"""Paired receiver decoding across upstream latent compute levels."""

import contextlib
import traceback
from copy import copy
from time import perf_counter
from typing import Any

from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.paths.naming import build_experiment_relpath
from ..infrastructure.paths.resolver import get_path_resolver
from ..infrastructure.storage import is_condition_complete
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import (
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
        args.receiver_reasoning = True
        exp_relpath = getattr(args, "exp_relpath", None) or build_experiment_relpath(
            args, dtype_name=str(config.get("model_dtype", ""))
        )
        directory = get_path_resolver().get_experiment_dir(exp_relpath)
        records_path = directory / "sample_results.jsonl"
        summary_path = directory / "summary.json"
        config_path = directory / "resolved_config.yaml"
        write_yaml(config_path, config)

        resolver = get_path_resolver()
        required_conditions = [
            f"u{s}_{m}" for s in [0, *steps] for m in ("answer_only", "free")
        ]
        all_complete = (
            not getattr(args, "overwrite", False)
            and summary_path.is_file()
            and len(items) > 0
        )
        if all_complete:
            for idx in range(len(items)):
                for cond in required_conditions:
                    if not is_condition_complete(
                        resolver.get_condition_dir(directory, idx, cond)
                    ):
                        all_complete = False
                        break
                if not all_complete:
                    break

        if all_complete:
            import json

            with contextlib.suppress(Exception):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                records = []
                for idx in range(len(items)):
                    for cond in required_conditions:
                        rf = (
                            resolver.get_condition_dir(directory, idx, cond)
                            / "result.json"
                        )
                        if rf.is_file():
                            records.append(json.loads(rf.read_text(encoding="utf-8")))
                if not records and records_path.is_file():
                    records = [
                        json.loads(line)
                        for line in records_path.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    ]
                print(f"Reusing completed receiver reasoning at {exp_relpath}")
                return summary, records

        tracker = self.tracker_port
        if tracker:
            run_name = f"{exp_relpath}_n{len(items)}"
            tags = {
                "experiment_type": "receiver_reasoning",
                "git_commit": config["git_commit"],
                "data_dir": str(directory.resolve()),
                "rel_dir": exp_relpath,
                "run_name": run_name,
            }
            if getattr(args, "version_tag", None):
                tags["version_tag"] = args.version_tag
            tracker.start_run(
                experiment_name=args.tracking_experiment_name,
                run_name=run_name,
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
