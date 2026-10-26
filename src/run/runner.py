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
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
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
    exp_rel = build_experiment_relpath(args, backend=backend)
    exp_dir = get_path_resolver().get_experiment_dir(exp_rel)
    summary_path = exp_dir / "summary.json"
    if not summary_path.is_file():
        summary_path = exp_dir / "collection_summary.json"
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    records = []
    samples_dir = exp_dir / "samples"
    if samples_dir.is_dir():
        for sdir in sorted(samples_dir.iterdir()):
            for cond in ("benchmark", "own", "result", "shared"):
                rf = sdir / cond / "result.json"
                if not rf.is_file():
                    rf = sdir / "result.json"
                if rf.is_file():
                    with contextlib.suppress(Exception):
                        records.append(json.loads(rf.read_text(encoding="utf-8")))
                    break
    if not records and (exp_dir / "sample_results.jsonl").is_file():
        with contextlib.suppress(Exception):
            for line in (
                (exp_dir / "sample_results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ):
                if line.strip():
                    records.append(json.loads(line))

    metrics = data.get("metrics", data)
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
