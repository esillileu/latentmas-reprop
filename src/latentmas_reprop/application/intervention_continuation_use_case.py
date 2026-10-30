"""Application use case for intervention continuation analysis."""

from typing import Any

import torch

from ..domain.ports.cache_port import CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.storage import FileStorage, get_file_storage


class InterventionContinuationUseCase:
    """Orchestrates shared latent calculation and intervention branches."""

    def __init__(
        self,
        dataset_port: DatasetPort | None = None,
        cache_port: CachePort | None = None,
        evaluator_port: EvaluatorPort | None = None,
        tracker_port: ExperimentTrackerPort | None = None,
        storage: FileStorage | None = None,
    ) -> None:
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR
        self.tracker_port = tracker_port
        self.storage = storage or get_file_storage()

    @torch.no_grad()
    def execute(
        self, model: ModelWrapper, args: Any
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        from .intervention_continuation.execution import execute

        return execute(self, model, args)
