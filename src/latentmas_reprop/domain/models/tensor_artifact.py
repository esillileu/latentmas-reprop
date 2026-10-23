"""Domain contracts for universal tensor storage and manifest metadata."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class StorageError(Exception):
    """Base exception for all storage and manifest operations."""


class ManifestCorruptedError(StorageError):
    """Raised when a manifest file exists on disk but is invalid or corrupted."""


class TensorMissingError(StorageError, FileNotFoundError):
    """Raised when a tensor referenced in the manifest does not exist on disk."""


class TensorChecksumMismatchError(StorageError, ValueError):
    """Raised when a stored tensor file fails SHA-256 integrity check."""


class ImmutableTensorOverwriteError(StorageError, FileExistsError):
    """Raised when attempting to overwrite an immutable persistent tensor."""


class RetentionClass(StrEnum):
    """Retention classification for stored tensor artifacts."""

    PERSISTENT = "persistent"  # Source of truth, long-term preservation
    REGENERABLE = "regenerable"  # Can be reconstructed by re-running deterministic step
    TEMPORARY = "temporary"  # Intermediate/scratch tensor, discarded post-experiment


class ArtifactStatus(StrEnum):
    """Integrity and availability status of a stored tensor artifact."""

    COMPLETE = "complete"  # Written atomically, checksum verified
    IN_PROGRESS = "in_progress"  # Write started but not yet finalized
    CORRUPTED = "corrupted"  # Checksum mismatch
    MISSING = "missing"  # Referenced in manifest but file does not exist on disk


@dataclass
class TensorEntry:
    """Individual tensor artifact record registered in manifest."""

    run_id: str
    sample_key: str
    branch_id: str
    segment_id: str
    artifact_id: str
    artifact_type: str
    storage_uri: str
    shape: list[int]
    dtype: str
    checksum: str
    schema_version: str = "1.0.0"
    retention_class: str = RetentionClass.PERSISTENT
    status: str = ArtifactStatus.COMPLETE
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def composite_key(self) -> str:
        """Composite identifier: sample_key:branch_id:segment_id:artifact_id."""
        s_key = self.sample_key or "run"
        b_id = self.branch_id or "main"
        seg_id = self.segment_id or "0"
        return f"{s_key}:{b_id}:{seg_id}:{self.artifact_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorEntry":
        valid_keys = cls.__dataclass_fields__.keys()
        clean = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**clean)


@dataclass
class TensorManifest:
    """Manifest describing all tensor artifacts associated with a run."""

    run_id: str
    schema_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    entries: dict[str, TensorEntry] = field(default_factory=dict)

    def add_entry(self, entry: TensorEntry) -> None:
        self.entries[entry.composite_key] = entry
        self.updated_at = datetime.now(UTC).isoformat()

    def get_entry(
        self,
        artifact_id: str,
        sample_key: str = "",
        branch_id: str = "",
        segment_id: str = "",
    ) -> TensorEntry | None:
        """Retrieve entry by composite coordinates or unique artifact_id."""
        # 1. Exact composite key lookup if coordinates provided
        if sample_key or branch_id or segment_id:
            s_key = sample_key or "run"
            b_id = branch_id or "main"
            seg_id = segment_id or "0"
            exact_key = f"{s_key}:{b_id}:{seg_id}:{artifact_id}"
            if exact_key in self.entries:
                return self.entries[exact_key]

        # 2. Filter matching entries
        matches = [
            e
            for e in self.entries.values()
            if e.artifact_id == artifact_id
            and (not sample_key or e.sample_key == sample_key)
            and (not branch_id or e.branch_id == branch_id)
            and (not segment_id or e.segment_id == segment_id)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous artifact query: found {len(matches)} entries matching artifact_id '{artifact_id}'. "
                "Specify sample_key, branch_id, and segment_id to disambiguate."
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": {k: v.to_dict() for k, v in self.entries.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TensorManifest":
        entries = {}
        for _k, v in data.get("entries", {}).items():
            entry = TensorEntry.from_dict(v)
            # Use composite_key as authoritative map key
            entries[entry.composite_key] = entry
        return cls(
            run_id=data.get("run_id", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            entries=entries,
        )
