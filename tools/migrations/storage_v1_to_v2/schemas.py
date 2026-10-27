"""Schemas and data models for Storage v1 to v2 migration tool."""

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class MigrationItemType(StrEnum):
    MLFLOW_DB = "mlflow_db"
    MLFLOW_ARTIFACT = "mlflow_artifact"
    BENCHMARK_RUN = "benchmark_run"
    ACQUISITION_RESULT = "acquisition_result"
    INTERVENTION_KV = "intervention_kv"
    TENSOR_BINARY = "tensor_binary"
    OTHER = "other"


class MigrationStatus(StrEnum):
    PENDING = "pending"
    COPIED = "copied"
    SKIPPED = "skipped"
    FAILED = "failed"
    ORPHAN = "orphan"


@dataclass
class InventoryItem:
    """Description of an identified legacy data artifact."""

    rel_path: str
    abs_path: str
    item_type: str
    size_bytes: int
    run_id: str | None = None
    checksum: str | None = None
    is_orphan: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InventoryItem":
        return cls(**data)


@dataclass
class InventoryReport:
    """Summary and details of all legacy items found during inventory scan."""

    source_root: str
    scanned_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_files: int = 0
    total_bytes: int = 0
    run_count: int = 0
    orphan_count: int = 0
    items: list[InventoryItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_root": self.source_root,
            "scanned_at": self.scanned_at,
            "total_files": self.total_files,
            "total_bytes": self.total_bytes,
            "run_count": self.run_count,
            "orphan_count": self.orphan_count,
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InventoryReport":
        items = [InventoryItem.from_dict(i) for i in data.get("items", [])]
        return cls(
            source_root=data.get("source_root", ""),
            scanned_at=data.get("scanned_at", ""),
            total_files=data.get("total_files", 0),
            total_bytes=data.get("total_bytes", 0),
            run_count=data.get("run_count", 0),
            orphan_count=data.get("orphan_count", 0),
            items=items,
        )


@dataclass
class MigrationRecord:
    """Record of a migration action on a single file/artifact."""

    source_path: str
    destination_path: str
    item_type: str
    run_id: str | None
    checksum_source: str
    checksum_dest: str | None
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationReport:
    """Result of validating migrated destination against source."""

    source_root: str
    destination_root: str
    verified_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    total_checked: int = 0
    success_count: int = 0
    mismatch_count: int = 0
    missing_count: int = 0
    db_valid: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.mismatch_count == 0
            and self.missing_count == 0
            and self.db_valid
            and len(self.errors) == 0
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
