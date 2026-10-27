import contextlib
import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.application.receiver_acquisition_use_case import (
    ReceiverAcquisitionUseCase,
)
from latentmas_reprop.application.receiver_compute_preflight_use_case import (
    ReceiverComputePreflightUseCase,
)
from latentmas_reprop.application.receiver_reasoning_use_case import (
    ReceiverReasoningUseCase,
)
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
from latentmas_reprop.infrastructure.paths.naming import (
    build_experiment_relpath,
    resolve_experiment_dtype_name,
)
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
from latentmas_reprop.infrastructure.storage.tensor_store import is_condition_complete
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def auto_device(device: str | None = None) -> torch.device:
    """Resolve target execution device."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def finalize_run_args(args: Any) -> None:
    """Ensure all decisive calculation options are fixed before model loading and path resolution."""
    if getattr(args, "receiver_reasoning", False):
        args.temperature = 0.0
        args.top_p = 1.0
        if getattr(args, "handoff_positions", None) is not None and args.handoff_positions <= 0:
            raise ValueError("Handoff positions must be positive")
        args.handoff_mode = "full" if getattr(args, "handoff_positions", None) is None else "tail"
        args.include_no_handoff = True
    elif getattr(args, "receiver_compute_preflight", False):
        args.temperature = 0.0
        args.top_p = 1.0


def _reuse_completed_run(args: Any) -> tuple[dict, list[dict]] | None:
    """Return finished experiment results without loading the model if completely done."""
    if getattr(args, "overwrite", False):
        return None
    backend = "vllm" if getattr(args, "use_vllm", False) else "transformers"
    dtype_name = resolve_experiment_dtype_name(args)
    exp_rel = build_experiment_relpath(args, dtype_name=dtype_name, backend=backend)
    resolver = get_path_resolver()
    exp_dir = resolver.get_experiment_dir(exp_rel)

    summary_path = exp_dir / "summary.json"
    if not summary_path.is_file() and (exp_dir / "collection_summary.json").is_file():
        summary_path = exp_dir / "collection_summary.json"

    # Determine required conditions and target count
    if getattr(args, "receiver_compute_preflight", False):
        max_s = int(getattr(args, "max_samples", 0))
        upstream = getattr(args, "upstream_steps", [20])
        conditions = ["no_handoff"] + [f"u{s}_matched" for s in upstream] + [f"u{s}_mismatched" for s in upstream]
    elif getattr(args, "receiver_reasoning", False):
        max_s = int(getattr(args, "max_samples", 0))
        upstream = [0, *getattr(args, "upstream_steps", [20])]
        conditions = [f"u{s}_{m}" for s in upstream for m in ("answer_only", "free")]
    elif getattr(args, "acquisition", False):
        max_s = int(getattr(args, "max_samples", 100))
        modes = list(getattr(args, "context_modes", ["full", "latent_only", "prompt_only"]))
        conds = list(getattr(args, "acquisition_conditions", ["own", "cross", "drop"]))
        conditions = [c if c in {"drop", "drop_position_matched"} else f"{m}_{c}" for c in conds for m in ([""] if c in {"drop", "drop_position_matched"} else modes)]
    elif getattr(args, "intervention", False):
        max_s = int(getattr(args, "max_samples", 0))
        raw_c = getattr(args, "intervention_conditions", ["own", "cross", "drop", "zero"])
        conditions = [c.strip() for c in raw_c.split(",")] if isinstance(raw_c, str) else list(raw_c)
    else:
        max_s = int(getattr(args, "max_samples", 0))
        conditions = ["benchmark"]

    if max_s > 0:
        target_count = max_s
    else:
        try:
            from latentmas_reprop.infrastructure.datasets.registry import (
                DEFAULT_DATASET_REGISTRY,
            )

            dataset = list(
                DEFAULT_DATASET_REGISTRY.load(
                    task=getattr(args, "task", "gsm8k"),
                    split=getattr(args, "split", "test"),
                )
            )
            target_count = len(dataset)
        except Exception:
            samples_dir = exp_dir / "samples"
            if samples_dir.is_dir():
                target_count = len([d for d in samples_dir.iterdir() if d.is_dir()])
            else:
                return None
        if target_count <= 0:
            return None

    # Verify every required sample and condition is complete on disk
    for idx in range(target_count):
        for cond in conditions:
            cond_dir = resolver.get_condition_dir(exp_dir, idx, cond)
            if not is_condition_complete(cond_dir):
                if cond == "benchmark" and is_condition_complete(
                    exp_dir / "samples" / f"{idx:06d}"
                ):
                    continue
                sdir = exp_dir / "samples" / f"{idx:06d}"
                if sdir.is_dir() and any(
                    is_condition_complete(c)
                    for c in sdir.iterdir()
                    if c.name == cond or c.name.startswith(f"{cond}_")
                ):
                    continue
                return None

    # Read records from disk
    records = []
    for idx in range(target_count):
        for cond in conditions:
            rf = resolver.get_condition_dir(exp_dir, idx, cond) / "result.json"
            if not rf.is_file() and cond == "benchmark":
                rf = exp_dir / "samples" / f"{idx:06d}" / "result.json"
            if not rf.is_file():
                sdir = exp_dir / "samples" / f"{idx:06d}"
                if sdir.is_dir():
                    candidates = [
                        c / "result.json"
                        for c in sdir.iterdir()
                        if (c.name == cond or c.name.startswith(f"{cond}_"))
                        and (c / "result.json").is_file()
                    ]
                    if candidates:
                        rf = candidates[0]
            if rf.is_file():
                with contextlib.suppress(Exception):
                    records.append(json.loads(rf.read_text(encoding="utf-8")))

    # Read or synthesize summary
    summary_data = {}
    if summary_path.is_file():
        with contextlib.suppress(Exception):
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    s_count = summary_data.get("sample_count") or summary_data.get("metrics", {}).get("total")
    if not summary_data or s_count != target_count:
        if getattr(args, "receiver_reasoning", False):
            from latentmas_reprop.application.receiver_reasoning.metrics import (
                summarize_receiver_records,
            )

            steps = getattr(args, "upstream_steps", [20])
            agg = summarize_receiver_records(records, steps)
            summary_data = agg | {"sample_count": target_count, "config": summary_data.get("config", {})}
        else:
            correct = sum(1 for r in records if r.get("correct", False))
            acc = correct / max(1, len(records))
            metrics_dict = dict(summary_data.get("metrics", {}))
            metrics_dict.update({
                "accuracy": acc, "correct": correct, "total": len(records), "max_samples": len(records),
                "sample_count": len(records),
                "requested_trajectories": target_count, "reused_trajectories": len(records),
                "fresh_trajectories": 0, "failed_trajectories": 0,
            })
            summary_data = {
                "status": "complete", "sample_count": len(records),
                "requested_trajectories": target_count, "reused_trajectories": len(records),
                "fresh_trajectories": 0, "failed_trajectories": 0,
                "metrics": metrics_dict, "config": summary_data.get("config", {}),
            }
        if not summary_path.is_file():
            with contextlib.suppress(Exception):
                summary_path.write_text(json.dumps(summary_data, indent=2) + "\n", encoding="utf-8")

    # Log to MLflow if tracking experiment is specified
    exp_name = getattr(args, "tracking_experiment_name", None)
    if exp_name:
        with contextlib.suppress(Exception):
            tracker = MLflowTracker()
            scope_str = f"n{target_count}"
            run_name = f"{exp_rel}_{scope_str}"
            tracker.start_run(
                experiment_name=exp_name,
                run_name=run_name,
                tags={
                    "data_dir": str(exp_dir.resolve()),
                    "rel_dir": exp_rel,
                    "run_name": run_name,
                },
            )
            tracker.log_dict(summary_data, "summary.json")
            tracker.end_run("FINISHED")

    metrics = summary_data.get("metrics", summary_data)
    print(f"Reusing completed experiment at {exp_rel}")
    if getattr(args, "print_summary", False):
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics, records


def run_benchmark(args: Any) -> tuple[dict, list[dict]]:
    """Execute a benchmark experiment with given parsed arguments."""
    finalize_run_args(args)
    exp_name = getattr(args, "tracking_experiment_name", None)
    if not exp_name:
        raise ValueError(
            "MLflow tracking experiment name must be specified (--tracking_experiment_name or via config) before running model inference."
        )

    set_seed(args.seed)
    device = auto_device(args.device)

    backend = "vllm" if getattr(args, "use_vllm", False) else "transformers"
    dtype_name = resolve_experiment_dtype_name(args)
    exp_relpath = build_experiment_relpath(args, dtype_name=dtype_name, backend=backend)
    args.exp_relpath = exp_relpath
    args.run_id = exp_relpath

    reused = _reuse_completed_run(args)
    if reused is not None:
        return reused

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    model.load_time_sec = time.perf_counter() - load_started

    if getattr(args, "receiver_compute_preflight", False):
        return ReceiverComputePreflightUseCase(tracker_port=MLflowTracker()).execute(
            model, args
        )

    if getattr(args, "receiver_reasoning", False):
        summary, records = ReceiverReasoningUseCase(
            tracker_port=MLflowTracker()
        ).execute(model, args)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary, records

    if getattr(args, "acquisition", False):
        tracker = MLflowTracker()
        use_case = ReceiverAcquisitionUseCase(tracker_port=tracker)
        metrics, records = use_case.execute(model, args)
        metrics_dict = metrics.to_dict()
        print("\n================ Receiver Acquisition Summary ================")
        print(json.dumps(metrics_dict, ensure_ascii=False, indent=2))
        return metrics_dict, [record.to_dict() for record in records]

    if getattr(args, "intervention", False):
        tracker = MLflowTracker()
        intervention_use_case = InterventionUseCase(tracker_port=tracker)
        metrics, records = intervention_use_case.execute(model, args)
        metrics_dict = metrics.to_dict()
        print("\n==================== Intervention Summary ====================")
        print(json.dumps(metrics_dict, ensure_ascii=False, indent=2))
        return metrics_dict, [r.to_dict() for r in records]

    tracker = MLflowTracker()
    use_case = BenchmarkUseCase(tracker_port=tracker)

    metrics, preds = use_case.execute(model, args)

    metrics_dict = metrics.to_dict()
    if getattr(args, "print_summary", False):
        print(json.dumps(metrics_dict, ensure_ascii=False))

    return metrics_dict, preds
