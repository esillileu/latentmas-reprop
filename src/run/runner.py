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
    if not summary_path.is_file():
        summary_path = exp_dir / "collection_summary.json"

    # Determine required conditions and target count
    if getattr(args, "receiver_compute_preflight", False):
        max_s = int(getattr(args, "max_samples", 0))
        upstream = getattr(args, "upstream_steps", [20])
        conditions = (
            ["no_handoff"]
            + [f"u{s}_matched" for s in upstream]
            + [f"u{s}_mismatched" for s in upstream]
        )
    elif getattr(args, "receiver_reasoning", False):
        max_s = int(getattr(args, "max_samples", 0))
        upstream = [0, *getattr(args, "upstream_steps", [20])]
        conditions = [f"u{s}_{m}" for s in upstream for m in ("answer_only", "free")]
    elif getattr(args, "acquisition", False):
        max_s = int(getattr(args, "max_samples", 100))
        modes = list(
            getattr(args, "context_modes", ["full", "latent_only", "prompt_only"])
        )
        conds = list(getattr(args, "acquisition_conditions", ["own", "cross", "drop"]))
        conditions = []
        for c in conds:
            if c in {"drop", "drop_position_matched"}:
                conditions.append(c)
            else:
                for m in modes:
                    conditions.append(f"{m}_{c}")
    elif getattr(args, "intervention", False):
        max_s = int(getattr(args, "max_samples", 0))
        raw_c = getattr(
            args, "intervention_conditions", ["own", "cross", "drop", "zero"]
        )
        conditions = (
            [c.strip() for c in raw_c.split(",")]
            if isinstance(raw_c, str)
            else list(raw_c)
        )
    else:
        max_s = int(getattr(args, "max_samples", 0))
        conditions = ["benchmark"]

    if max_s > 0:
        target_count = max_s
        if summary_path.is_file():
            try:
                sdata = json.loads(summary_path.read_text(encoding="utf-8"))
                if (
                    isinstance(sdata, dict)
                    and sdata.get("sample_count", 0) < target_count
                ):
                    return None
            except Exception:
                return None
    else:
        if not summary_path.is_file():
            return None
        try:
            sdata = json.loads(summary_path.read_text(encoding="utf-8"))
            if not isinstance(sdata, dict):
                return None
            cfg = sdata.get("config", {})
            if cfg.get("requested_max_samples") not in (-1, None, 0):
                return None
            target_count = int(sdata.get("sample_count", 0))
            if target_count <= 0:
                return None
        except Exception:
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
                return None

    # Read summary and records
    summary_data = {}
    if summary_path.is_file():
        with contextlib.suppress(Exception):
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary_data.get("metrics", summary_data)

    records = []
    for idx in range(target_count):
        for cond in conditions:
            rf = resolver.get_condition_dir(exp_dir, idx, cond) / "result.json"
            if not rf.is_file() and cond == "benchmark":
                rf = exp_dir / "samples" / f"{idx:06d}" / "result.json"
            if rf.is_file():
                with contextlib.suppress(Exception):
                    records.append(json.loads(rf.read_text(encoding="utf-8")))

    print(f"Reusing completed experiment at {exp_rel}")
    if getattr(args, "print_summary", False):
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics, records


def run_benchmark(args: Any) -> tuple[dict, list[dict]]:
    """Execute a benchmark experiment with given parsed arguments."""
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
