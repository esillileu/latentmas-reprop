import json
import os
import random
from typing import Any

import numpy as np
import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.application.receiver_acquisition_use_case import (
    ReceiverAcquisitionUseCase,
)
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


def run_benchmark(args: Any) -> tuple[dict, list[dict]]:
    """Execute a benchmark experiment with given parsed arguments."""
    set_seed(args.seed)
    device = auto_device(args.device)

    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

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
