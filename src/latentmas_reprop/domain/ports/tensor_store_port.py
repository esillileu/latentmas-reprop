"""Port interface for file-based tensor storage."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch


class TensorStorePort(ABC):
    """Abstract port for persistent tensor storage."""

    @abstractmethod
    def save_tensor(
        self,
        path: Path | str,
        tensor: torch.Tensor,
        overwrite: bool = False,
    ) -> Path:
        """Atomically persist a tensor to file."""
        raise NotImplementedError

    @abstractmethod
    def load_tensor(
        self,
        path: Path | str,
        map_location: Any = None,
    ) -> torch.Tensor:
        """Load a tensor from file."""
        raise NotImplementedError
