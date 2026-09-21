from abc import ABC, abstractmethod
from enum import StrEnum
from pathlib import Path
from typing import Any


class CacheLayer(StrEnum):
    """Hierarchical usage layers for repository execution caches in .cache/."""

    MODELS = "models"
    MODELS_REALIGN = "models/realign"
    EVALUATION = "evaluation"
    EVALUATION_RUNS = "evaluation/runs"
    DATASETS = "datasets"
    RUNTIME = "runtime"


class CachePort(ABC):
    """Port interface for hierarchical execution cache management."""

    @abstractmethod
    def get_layer_path(self, layer: CacheLayer | str) -> Path:
        """Get the filesystem path for the specified cache layer."""
        raise NotImplementedError

    @abstractmethod
    def save_json(self, layer: CacheLayer | str, filename: str, data: Any) -> Path:
        """Save JSON data to the specified cache layer."""
        raise NotImplementedError

    @abstractmethod
    def load_json(self, layer: CacheLayer | str, filename: str) -> Any:
        """Load JSON data from the specified cache layer."""
        raise NotImplementedError

    @abstractmethod
    def save_torch(self, layer: CacheLayer | str, filename: str, obj: Any) -> Path:
        """Save PyTorch tensor/model object to the specified cache layer."""
        raise NotImplementedError

    @abstractmethod
    def load_torch(
        self, layer: CacheLayer | str, filename: str, map_location: Any = None
    ) -> Any:
        """Load PyTorch tensor/model object from the specified cache layer."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, layer: CacheLayer | str, filename: str) -> bool:
        """Check if an item exists in the specified cache layer."""
        raise NotImplementedError
