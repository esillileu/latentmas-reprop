"""Paired receiver compute curves with a single trajectory per handoff."""

import traceback
from copy import copy

from ..domain.models import compute_sample_key
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.paths.naming import build_experiment_relpath
from ..infrastructure.paths.resolver import get_path_resolver
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import clean_config_args, write_json, write_jsonl, write_yaml
from .receiver_compute_preflight.collection import collect_samples
from .receiver_compute_preflight.tracing import run_identity
from .receiver_reasoning.runtime import RuntimeMeasurements


class ReceiverComputePreflightUseCase:
    def __init__(self, dataset_port=None, evaluator_port=None, tracker_port=None):
        self.dataset = dataset_port or DEFAULT_DATASET_REGISTRY
        self.evaluator = evaluator_port or DEFAULT_EVALUATOR
        self.tracker = tracker_port

    def execute(self, model, args):
        if self.tracker is None:
            raise ValueError("Receiver compute preflight requires MLflow tracking")
        if args.method != "latent_mas" or args.use_vllm or args.task != "gsm8k":
            raise ValueError("Preflight requires GSM8K transformers LatentMAS")
        if (
            args.handoff_positions is not None
            or getattr(args, "latent_only", False)
            or getattr(args, "sequential_info_only", False)
        ):
            raise ValueError("Preflight requires full KV handoff without truncation")
        args = copy(args)
        args.temperature, args.top_p = 0.0, 1.0
        items = list(self.dataset.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            items = items[: args.max_samples]
        ids = [compute_sample_key(args.task, args.split, i["question"]) for i in items]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Preflight requires at least two unique paired samples")
        config = clean_config_args(args) | {
            "sample_ids": ids,
            "model_dtype": getattr(model, "dtype_name", None),
            "git_commit": get_git_commit_hash(),
            "handoff_mode": "full",
            "receiver_mode": "free",
            "donor_policy": "minimum_total_absolute_length_derangement",
            "answer_policy": "explicit_complete_final_answer",
            "receiver_latency_definition": "synchronized generation including receiver prefill; excludes cache transfer, evaluation and discarded retries",
            "compute_proxy_definition": "causal attention pairs per layer/head and processed transformer positions; not FLOPs",
        }
        args.receiver_compute_preflight = True
        exp_relpath = build_experiment_relpath(
            args, dtype_name=str(config.get("model_dtype", ""))
        )
        directory = get_path_resolver().get_experiment_dir(exp_relpath)
        summary_file = directory / "collection_summary.json"
        records_file = directory / "sample_results.jsonl"
        config_file = directory / "resolved_config.yaml"
        write_yaml(config_file, config)

        if not getattr(args, "overwrite", False) and summary_file.is_file():
            import json

            try:
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                records = []
                if records_file.is_file():
                    records = [
                        json.loads(line)
                        for line in records_file.read_text(
                            encoding="utf-8"
                        ).splitlines()
                        if line.strip()
                    ]
                print(f"Reusing completed preflight at {exp_relpath}")
                return summary, records
            except Exception:
                pass

        tracker = self.tracker
        _name, kind, label = run_identity(args, len(items))
        config.update(
            run_name=exp_relpath,
            run_kind=kind,
            run_label=label,
            version_tag=args.version_tag or "receiver-compute-preflight",
        )
        tracker.start_run(
            experiment_name=args.tracking_experiment_name,
            run_name=exp_relpath,
            tags={
                k: config[k]
                for k in (
                    "git_commit",
                    "version_tag",
                    "run_kind",
                    "run_label",
                    "model_name",
                    "task",
                    "split",
                    "seed",
                )
            }
            | {
                "experiment_type": "receiver_compute_preflight",
                "data_dir": str(directory.resolve()),
                "rel_dir": exp_relpath,
            },
        )
        config["run_id"] = exp_relpath
        tracker.log_params(config)
        manifest = []
        records = []
        runtime = RuntimeMeasurements(model)
        if True:
            try:
                collect_samples(
                    model,
                    args,
                    items,
                    ids,
                    config,
                    tracker,
                    self.evaluator,
                    runtime,
                    records,
                    manifest,
                    records_file,
                )
                summary = {
                    "sample_count": len(items),
                    "result_count": len(records),
                    "trace_count": len(manifest),
                    "run_id": exp_relpath,
                    "runtime": runtime.run_peak(),
                }
                write_json(summary_file, summary)
            except (Exception, KeyboardInterrupt):
                write_json(
                    directory / "failure.json", {"traceback": traceback.format_exc()}
                )
                write_json(directory / "trace_manifest.json", manifest)
                write_jsonl(directory / "sample_results.jsonl", records)
                for path in directory.iterdir():
                    tracker.log_artifact(path, artifact_path="results")
                tracker.flush_traces()
                tracker.end_run(status="FAILED")
                raise
            write_json(directory / "trace_manifest.json", manifest)
            for path in directory.iterdir():
                tracker.log_artifact(path, artifact_path="results")
            tracker.flush_traces()
            tracker.end_run(status="FINISHED")
        return summary, records
