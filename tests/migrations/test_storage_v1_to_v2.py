"""Tests for standalone Storage v1 to v2 migration tool."""

import sqlite3
from pathlib import Path

import pytest
import torch

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage.reader import ResearchDataReader
from latentmas_reprop.infrastructure.storage.tensor_store import TensorStore
from tools.migrations.storage_v1_to_v2.inventory import generate_inventory
from tools.migrations.storage_v1_to_v2.migrate import StorageMigrator, compute_sha256
from tools.migrations.storage_v1_to_v2.schemas import MigrationStatus
from tools.migrations.storage_v1_to_v2.verify import MigrationVerifier


@pytest.fixture
def synthetic_legacy_v1(tmp_path: Path) -> Path:
    """Create a realistic synthetic legacy v1 directory structure."""
    legacy_root = tmp_path / "legacy_cache"
    legacy_root.mkdir(parents=True, exist_ok=True)

    # 1. Legacy SQLite DB with 2 runs
    db_path = legacy_root / "mlflow.db"
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE runs (run_uuid TEXT PRIMARY KEY, experiment_id TEXT, name TEXT, status TEXT, artifact_uri TEXT)"
    )
    cur.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        (
            "run_alpha_123",
            "0",
            "alpha_exp",
            "FINISHED",
            f"file://{legacy_root}/mlflow_artifacts/0/run_alpha_123/artifacts",
        ),
    )
    cur.execute(
        "INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
        (
            "run_beta_456",
            "0",
            "beta_exp",
            "FINISHED",
            f"file://{legacy_root}/mlflow_artifacts/0/run_beta_456/artifacts",
        ),
    )
    conn.commit()
    conn.close()

    # 2. Legacy MLflow Artifacts
    artifact_a = (
        legacy_root
        / "mlflow_artifacts"
        / "0"
        / "run_alpha_123"
        / "artifacts"
        / "results"
        / "summary.json"
    )
    artifact_a.parent.mkdir(parents=True, exist_ok=True)
    artifact_a.write_text(
        '{"run_id": "run_alpha_123", "accuracy": 0.85}', encoding="utf-8"
    )

    artifact_b = (
        legacy_root
        / "mlflow_artifacts"
        / "0"
        / "run_beta_456"
        / "artifacts"
        / "results"
        / "summary.json"
    )
    artifact_b.parent.mkdir(parents=True, exist_ok=True)
    artifact_b.write_text(
        '{"run_id": "run_beta_456", "accuracy": 0.92}', encoding="utf-8"
    )

    # 3. Legacy Benchmark run
    bench_dir = legacy_root / "evaluation" / "runs" / "run_alpha_123"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "samples.jsonl").write_text(
        '{"sample_key": "k1", "answer": 1}\n', encoding="utf-8"
    )
    (bench_dir / "summary.json").write_text(
        '{"status": "complete", "sample_count": 1}', encoding="utf-8"
    )

    # 4. Legacy tensor binaries
    tensor_file_a = (
        legacy_root
        / "evaluation"
        / "receiver_acquisition"
        / "run_alpha_123_sender_latent_states.pt"
    )
    tensor_file_a.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor([1.0, 2.0, 3.0]), tensor_file_a)

    # 5. Legacy orphan tensor
    orphan_tensor = (
        legacy_root / "evaluation" / "receiver_acquisition" / "sender_latent_states.pt"
    )
    torch.save(torch.tensor([4.0, 5.0, 6.0]), orphan_tensor)

    return legacy_root


def test_inventory_scan(synthetic_legacy_v1: Path):
    report = generate_inventory(synthetic_legacy_v1)
    assert report.total_files > 0
    assert report.run_count >= 2
    assert report.orphan_count >= 1

    db_items = [i for i in report.items if i.item_type == "mlflow_db"]
    assert len(db_items) == 1


def test_dry_run_migration_does_not_modify_disk(
    synthetic_legacy_v1: Path, tmp_path: Path
):
    dest_root = tmp_path / "new_storage"
    migrator = StorageMigrator(
        source_root=synthetic_legacy_v1, dest_root=dest_root, dry_run=True
    )
    records = migrator.run()

    assert len(records) > 0
    assert all(r.status == MigrationStatus.PENDING for r in records)
    assert not dest_root.exists(), (
        "Dry-run must not create destination directory or files"
    )


def test_migration_execution_and_verification(
    synthetic_legacy_v1: Path, tmp_path: Path, monkeypatch
):
    dest_root = tmp_path / "new_storage"
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(dest_root))

    # Compute source checksum snapshot before migration
    src_hashes_before = {
        p: compute_sha256(p) for p in synthetic_legacy_v1.rglob("*") if p.is_file()
    }

    migrator = StorageMigrator(
        source_root=synthetic_legacy_v1, dest_root=dest_root, dry_run=False
    )
    records = migrator.run()

    assert any(r.status == MigrationStatus.COPIED for r in records)
    assert not any(r.status == MigrationStatus.FAILED for r in records)

    # 1. Source files must remain 100% untouched
    src_hashes_after = {
        p: compute_sha256(p) for p in synthetic_legacy_v1.rglob("*") if p.is_file()
    }
    assert src_hashes_before == src_hashes_after, "Source files must not be altered"

    # 2. Run verification tool
    verifier = MigrationVerifier(source_root=synthetic_legacy_v1, dest_root=dest_root)
    v_report = verifier.verify()
    assert v_report.is_valid, f"Verification failed with errors: {v_report.errors}"
    assert v_report.db_valid
    assert v_report.mismatch_count == 0
    assert v_report.missing_count == 0

    # 3. Check SQLite DB in destination has updated artifact URIs
    dest_db = dest_root / "mlflow" / "mlflow.db"
    assert dest_db.is_file()
    conn = sqlite3.connect(str(dest_db))
    cur = conn.cursor()
    cur.execute("SELECT artifact_uri FROM runs WHERE run_uuid = 'run_alpha_123'")
    uri = cur.fetchone()[0]
    conn.close()
    assert str(dest_root / "mlflow" / "mlflow_artifacts") in uri

    # 4. Check ResearchDataReader reads migrated artifacts and tensors
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)
    reader = ResearchDataReader(path_resolver=resolver, tensor_store=store)

    # Load migrated tensor
    t = reader.load_tensor("run_alpha_123", "run_alpha_123_sender_latent_states")
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))

    # Load migrated benchmark artifact
    summary = reader.load_artifact("run_alpha_123", "summary.json")
    assert summary["status"] == "complete"


def test_idempotent_reexecution(synthetic_legacy_v1: Path, tmp_path: Path):
    dest_root = tmp_path / "new_storage"
    migrator = StorageMigrator(
        source_root=synthetic_legacy_v1, dest_root=dest_root, dry_run=False
    )
    migrator.run()

    # Second execution must skip already migrated files
    migrator_again = StorageMigrator(
        source_root=synthetic_legacy_v1, dest_root=dest_root, dry_run=False
    )
    records_again = migrator_again.run()

    assert all(r.status == MigrationStatus.SKIPPED for r in records_again)


def test_verification_detects_corruption(synthetic_legacy_v1: Path, tmp_path: Path):
    dest_root = tmp_path / "new_storage"
    migrator = StorageMigrator(
        source_root=synthetic_legacy_v1, dest_root=dest_root, dry_run=False
    )
    migrator.run()

    # Deliberately tamper with one migrated file
    tampered_file = next(dest_root.rglob("*.jsonl"))
    tampered_file.write_text("corrupted content", encoding="utf-8")

    verifier = MigrationVerifier(source_root=synthetic_legacy_v1, dest_root=dest_root)
    v_report = verifier.verify()
    assert not v_report.is_valid
    assert v_report.mismatch_count >= 1
