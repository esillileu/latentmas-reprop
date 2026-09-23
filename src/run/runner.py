import json
import os
import random
import time
from typing import Any

import numpy as np
import torch

from latentmas_reprop.application.benchmark_results import (
    BenchmarkRunStore,
    build_run_id,
)
from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.application.receiver_acquisition_use_case import (
    ReceiverAcquisitionUseCase,
)
from latentmas_reprop.infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from latentmas_reprop.infrastructure.models.dtype import dtype_name, resolve_model_dtype
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
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


def _reuse_completed_benchmark(
    args: Any, device: torch.device
) -> tuple[dict, list[dict]] | None:
    """Return a finished benchmark without loading the model again."""
    if getattr(args, "acquisition", False) or getattr(args, "intervention", False):
        return None
    selected = resolve_model_dtype(device)
    run_id = build_run_id(args, dtype_name(selected))
    store = BenchmarkRunStore(DEFAULT_CACHE_MANAGER, run_id)
    if not store.is_complete():
        return None
    metrics, preds = store.load_finished()
    print(f"Reusing completed run {run_id}")
    print(json.dumps(metrics.to_dict(), ensure_ascii=False))
    return metrics.to_dict(), preds


def run_benchmark(args: Any) -> tuple[dict, list[dict]]:
    """Execute a benchmark experiment with given parsed arguments."""
    set_seed(args.seed)
    device = auto_device(args.device)
    reused = _reuse_completed_benchmark(args, device)
    if reused is not None:
        return reused

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    model.load_time_sec = time.perf_counter() - load_started

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

    use_case = BenchmarkUseCase()

    metrics, preds = use_case.execute(model, args)

    metrics_dict = metrics.to_dict()
    # Output final summary JSON
    print(json.dumps(metrics_dict, ensure_ascii=False))

    return metrics_dict, preds
