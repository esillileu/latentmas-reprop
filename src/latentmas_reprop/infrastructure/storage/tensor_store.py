"""Universal persistent TensorStore implementation with manifest and checksum validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ...domain.models.tensor_artifact import (
    ArtifactStatus,
    RetentionClass,
    TensorEntry,
    TensorManifest,
)
from ...domain.ports.tensor_store_port import TensorStorePort
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ..paths.resolver import PathResolver, get_path_resolver


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


class TensorStore(TensorStorePort):
    """Universal repository tensor storage rooted at PathResolver.tensors_dir.

    Stores heavy tensors locally and generates lightweight manifests with SHA-256
    checksums that can be logged directly into MLflow artifacts without bloat.
    """

    def __init__(self, path_resolver: PathResolver | None = None) -> None:
        self._resolver = path_resolver or get_path_resolver()

    @property
    def base_dir(self) -> Path:
        return self._resolver.tensors_dir

    def _get_run_dir(self, run_id: str) -> Path:
        clean_run_id = str(run_id).replace("/", "_").replace("\\", "_")
        path = self.base_dir / clean_run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_manifest_path(self, run_id: str) -> Path:
        return self._get_run_dir(run_id) / "tensor_manifest.json"

    def get_manifest(self, run_id: str) -> TensorManifest | None:
        path = self._get_manifest_path(run_id)
        if not path.is_file():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return TensorManifest.from_dict(data)
        except Exception:
            return None

    def _save_manifest(self, manifest: TensorManifest) -> None:
        path = self._get_manifest_path(manifest.run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

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
        """Atomically persist a tensor with checksum and register it into run manifest."""
        run_dir = self._get_run_dir(run_id)
        safe_artifact_id = str(artifact_id).replace("/", "_").replace("\\", "_")
        target_file = run_dir / f"{safe_artifact_id}.pt"
        tmp_file = run_dir / f".{safe_artifact_id}.pt.tmp"

        # 1. Save tensor to temporary file
        torch.save(tensor, tmp_file)

        # 2. Compute checksum and shape
        checksum = compute_file_sha256(tmp_file)
        shape = list(tensor.shape) if hasattr(tensor, "shape") else []
        dtype = str(getattr(tensor, "dtype", "unknown"))

        # 3. Atomically replace target file
        tmp_file.replace(target_file)

        # 4. Construct entry
        storage_uri = f"file://{target_file.resolve().as_posix()}"
        entry = TensorEntry(
            run_id=run_id,
            sample_key=sample_key,
            branch_id=branch_id,
            segment_id=segment_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            storage_uri=storage_uri,
            shape=shape,
            dtype=dtype,
            checksum=checksum,
            retention_class=retention_class,
            status=ArtifactStatus.COMPLETE,
            metadata=metadata or {},
        )

        # 5. Atomically update manifest
        manifest = self.get_manifest(run_id) or TensorManifest(run_id=run_id)
        manifest.add_entry(entry)
        self._save_manifest(manifest)

        return entry

    def _resolve_entry_path(self, entry: TensorEntry) -> Path:
        uri = entry.storage_uri
        if uri.startswith("file://"):
            return Path(uri[7:])
        p = Path(uri)
        if p.is_absolute():
            return p
        return (self.base_dir / p).resolve()

    def load_tensor(
        self,
        entry_or_id: TensorEntry | str,
        run_id: str | None = None,
        verify_checksum: bool = True,
        map_location: Any = None,
    ) -> torch.Tensor:
        """Load a tensor artifact by entry or artifact_id, validating existence and checksum."""
        if isinstance(entry_or_id, TensorEntry):
            entry = entry_or_id
        else:
            if not run_id:
                raise ValueError("run_id must be provided when loading tensor by artifact_id string")
            entry = self.get_entry(run_id, entry_or_id)
            if entry is None:
                raise FileNotFoundError(f"Tensor artifact '{entry_or_id}' not found in manifest of run '{run_id}'")

        target_path = self._resolve_entry_path(entry)
        if not target_path.is_file():
            entry.status = ArtifactStatus.MISSING
            raise FileNotFoundError(f"Tensor file missing on disk: {target_path} (artifact: {entry.artifact_id})")

        if verify_checksum:
            current_checksum = compute_file_sha256(target_path)
            if current_checksum != entry.checksum:
                entry.status = ArtifactStatus.CORRUPTED
                raise ValueError(
                    f"Tensor artifact corrupted (checksum mismatch): {target_path}. "
                    f"Expected {entry.checksum}, found {current_checksum}"
                )

        return torch.load(target_path, map_location=map_location, weights_only=False)

    def get_entry(self, run_id: str, artifact_id: str) -> TensorEntry | None:
        manifest = self.get_manifest(run_id)
        if manifest is None:
            return None
        return manifest.get_entry(artifact_id)

    def register_reference(
        self,
        source_entry: TensorEntry,
        target_run_id: str,
        target_branch_id: str,
        target_artifact_id: str | None = None,
    ) -> TensorEntry:
        """Reference an existing tensor in another run/branch without copying the underlying tensor file."""
        new_artifact_id = target_artifact_id or source_entry.artifact_id
        ref_entry = TensorEntry(
            run_id=target_run_id,
            sample_key=source_entry.sample_key,
            branch_id=target_branch_id,
            segment_id=source_entry.segment_id,
            artifact_id=new_artifact_id,
            artifact_type=source_entry.artifact_type,
            storage_uri=source_entry.storage_uri,
            shape=source_entry.shape,
            dtype=source_entry.dtype,
            checksum=source_entry.checksum,
            retention_class=source_entry.retention_class,
            status=source_entry.status,
            metadata={
                **source_entry.metadata,
                "referenced_from_run_id": source_entry.run_id,
                "referenced_from_artifact_id": source_entry.artifact_id,
            },
        )
        manifest = self.get_manifest(target_run_id) or TensorManifest(run_id=target_run_id)
        manifest.add_entry(ref_entry)
        self._save_manifest(manifest)
        return ref_entry

    def sync_manifest_to_tracker(
        self, tracker_port: ExperimentTrackerPort, run_id: str
    ) -> None:
        """Register the lightweight tensor manifest in MLflow without uploading large tensor binaries."""
        manifest = self.get_manifest(run_id)
        if manifest:
            tracker_port.log_dict(manifest.to_dict(), "tensors/tensor_manifest.json")


DEFAULT_TENSOR_STORE = TensorStore()


def get_tensor_store() -> TensorStore:
    """Return the global default TensorStore instance."""
    return DEFAULT_TENSOR_STORE
