"""Tests for universal TensorStore, manifests, checksum validation, and ResearchDataReader."""

import pytest
import torch

from latentmas_reprop.domain.models.tensor_artifact import (
    ArtifactStatus,
    RetentionClass,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    ResearchDataReader,
    TensorStore,
)


def test_tensor_store_save_and_load(tmp_path):
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)

    tensor = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    entry = store.save_tensor(
        run_id="run_123",
        sample_key="sample_abc",
        artifact_id="hidden_state_step_10",
        tensor=tensor,
        artifact_type="hidden_state",
        branch_id="own",
        segment_id="1",
        retention_class=RetentionClass.PERSISTENT,
        metadata={"step": 10},
    )

    assert entry.run_id == "run_123"
    assert entry.artifact_id == "hidden_state_step_10"
    assert entry.shape == [2, 4, 16]
    assert entry.dtype == "torch.bfloat16"
    assert len(entry.checksum) == 64
    assert entry.status == ArtifactStatus.COMPLETE

    # Verify manifest was persisted
    manifest = store.get_manifest("run_123")
    assert manifest is not None
    assert "hidden_state_step_10" in manifest.entries

    # Load and verify equality
    loaded = store.load_tensor(entry)
    assert torch.equal(loaded, tensor)

    # Load by ID
    loaded_by_id = store.load_tensor("hidden_state_step_10", run_id="run_123")
    assert torch.equal(loaded_by_id, tensor)


def test_tensor_store_checksum_corruption_detection(tmp_path):
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)

    tensor = torch.zeros(4, 4, dtype=torch.float32)
    entry = store.save_tensor(
        run_id="run_corrupt",
        sample_key="sample_0",
        artifact_id="matrix",
        tensor=tensor,
        artifact_type="projection_matrix",
    )

    target_path = store._resolve_entry_path(entry)
    assert target_path.is_file()

    # Tamper with file
    with open(target_path, "r+b") as f:
        f.seek(10)
        f.write(b"\xff\xff\xff\xff")

    with pytest.raises(ValueError, match="checksum mismatch"):
        store.load_tensor(entry, verify_checksum=True)

    assert entry.status == ArtifactStatus.CORRUPTED


def test_tensor_store_missing_file_detection(tmp_path):
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)

    tensor = torch.ones(2, 2)
    entry = store.save_tensor(
        run_id="run_miss",
        sample_key="sample_1",
        artifact_id="kv_snapshot",
        tensor=tensor,
        artifact_type="kv_snapshot",
    )

    target_path = store._resolve_entry_path(entry)
    target_path.unlink()

    with pytest.raises(FileNotFoundError, match="Tensor file missing on disk"):
        store.load_tensor(entry)

    assert entry.status == ArtifactStatus.MISSING


def test_tensor_store_register_reference_no_duplication(tmp_path):
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)

    tensor = torch.randn(8, 8)
    source_entry = store.save_tensor(
        run_id="parent_run",
        sample_key="sample_parent",
        artifact_id="pre_branch_kv",
        tensor=tensor,
        artifact_type="kv_snapshot",
        branch_id="common",
    )

    # Branch references common snapshot
    branch_entry = store.register_reference(
        source_entry=source_entry,
        target_run_id="branch_run_1",
        target_branch_id="branch_a",
        target_artifact_id="input_kv_branch_a",
    )

    assert branch_entry.run_id == "branch_run_1"
    assert branch_entry.branch_id == "branch_a"
    assert branch_entry.storage_uri == source_entry.storage_uri
    assert branch_entry.checksum == source_entry.checksum

    # Verify loading from branch run
    loaded_branch = store.load_tensor("input_kv_branch_a", run_id="branch_run_1")
    assert torch.equal(loaded_branch, tensor)


def test_research_data_reader(tmp_path):
    resolver = PathResolver(tmp_path)
    store = TensorStore(resolver)
    reader = ResearchDataReader(path_resolver=resolver, tensor_store=store)

    # 1. Save tensor and load via reader
    tensor = torch.tensor([10.0, 20.0, 30.0])
    store.save_tensor(
        run_id="run_reader",
        sample_key="k1",
        artifact_id="test_tensor",
        tensor=tensor,
        artifact_type="test",
    )

    loaded_tensor = reader.load_tensor("run_reader", "test_tensor")
    assert torch.equal(loaded_tensor, tensor)

    # 2. Local JSON artifact loading
    sample_file = resolver.storage_dir / "run_reader" / "summary.json"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text('{"accuracy": 0.95}', encoding="utf-8")

    data = reader.load_artifact("run_reader", "summary.json")
    assert data == {"accuracy": 0.95}

    # 3. Non-existent artifact raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        reader.load_artifact("run_reader", "non_existent.json")

    # 4. Strict isolation: unscoped artifact from another run is blocked
    unscoped_summary = resolver.artifacts_dir / "other_run_summary.json"
    unscoped_summary.write_text('{"run_id": "different_run", "accuracy": 0.50}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Unscoped fallback was blocked"):
        reader.load_artifact("run_reader", "other_run_summary.json")

    # Config ID match must NOT be treated as valid run_id provenance
    config_only_summary = resolver.artifacts_dir / "config_match_summary.json"
    config_only_summary.write_text('{"config_id": "run_reader", "accuracy": 0.60}', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Unscoped fallback was blocked"):
        reader.load_artifact("run_reader", "config_match_summary.json")

    # Allowed when explicitly requested
    fallback_data = reader.load_artifact("run_reader", "other_run_summary.json", allow_unscoped_fallback=True)
    assert fallback_data["accuracy"] == 0.50

    # 5. Strict isolation: unscoped tensor is blocked unless allow_unscoped_fallback=True
    legacy_tensor_path = resolver.cache_dir / "evaluation" / "receiver_acquisition" / "unscoped_tensor.pt"
    legacy_tensor_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.tensor([99.0]), legacy_tensor_path)

    with pytest.raises(FileNotFoundError, match="Unscoped fallback was blocked"):
        reader.load_tensor("run_reader", "unscoped_tensor")

    fallback_tensor = reader.load_tensor("run_reader", "unscoped_tensor", allow_unscoped_fallback=True)
    assert torch.equal(fallback_tensor, torch.tensor([99.0]))
