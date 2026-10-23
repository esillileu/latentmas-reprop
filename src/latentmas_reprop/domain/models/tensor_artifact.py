"""Domain contracts for universal tensor storage and manifest metadata."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class RetentionClass(StrEnum):
    """Retention classification for stored tensor artifacts."""

    PERSISTENT = "persistent"      # Source of truth, long-term preservation
    REGENERABLE = "regenerable"    # Can be reconstructed by re-running deterministic step
    TEMPORARY = "temporary"        # Intermediate/scratch tensor, discarded post-experiment


class ArtifactStatus(StrEnum):
    """Integrity and availability status of a stored tensor artifact."""

    COMPLETE = "complete"          # Written atomically, checksum verified
    IN_PROGRESS = "in_progress"    # Write started but not yet finalized
    CORRUPTED = "corrupted"        # Checksum mismatch
    MISSING = "missing"            # Referenced in manifest but file does not exist on disk


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
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

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
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    entries: dict[str, TensorEntry] = field(default_factory=dict)

    def add_entry(self, entry: TensorEntry) -> None:
        self.entries[entry.artifact_id] = entry
        self.updated_at = datetime.now(UTC).isoformat()

    def get_entry(self, artifact_id: str) -> TensorEntry | None:
        return self.entries.get(artifact_id)

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
        entries = {
            k: TensorEntry.from_dict(v)
            for k, v in data.get("entries", {}).items()
        }
        return cls(
            run_id=data.get("run_id", ""),
            schema_version=data.get("schema_version", "1.0.0"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            entries=entries,
        )
