"""Port interface for universal tensor storage and manifest registry."""

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..models.tensor_artifact import (
    RetentionClass,
    TensorEntry,
    TensorManifest,
)


class TensorStorePort(ABC):
    """Abstract port for persistent tensor storage with checksum validation and manifest tracking."""

    @abstractmethod
    def save_tensor(
        self,
        run_id: str,
        sample_key: str,
        artifact_id: str,
        tensor: torch.Tensor,
        artifact_type: str,
        branch_id: str = "default",
        segment_id: str = "0",
        retention_class: str = RetentionClass.PERSISTENT,
        metadata: dict[str, Any] | None = None,
    ) -> TensorEntry:
        """Atomically persist a tensor, compute its checksum and shape, and register it in the run manifest."""
        raise NotImplementedError

    @abstractmethod
    def load_tensor(
        self,
        entry_or_id: TensorEntry | str,
        run_id: str | None = None,
        verify_checksum: bool = True,
        map_location: Any = None,
    ) -> torch.Tensor:
        """Load a tensor artifact by entry or artifact_id, validating existence and integrity."""
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, run_id: str, artifact_id: str) -> TensorEntry | None:
        """Retrieve the metadata entry for a tensor artifact from the run manifest."""
        raise NotImplementedError

    @abstractmethod
    def get_manifest(self, run_id: str) -> TensorManifest | None:
        """Load the manifest for a given run ID."""
        raise NotImplementedError

    @abstractmethod
    def register_reference(
        self,
        source_entry: TensorEntry,
        target_run_id: str,
        target_branch_id: str,
        target_artifact_id: str | None = None,
    ) -> TensorEntry:
        """Reference an existing tensor in another run/branch without physically duplicating the tensor file."""
        raise NotImplementedError
