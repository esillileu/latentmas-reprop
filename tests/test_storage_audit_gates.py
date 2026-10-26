"""Audit gate tests for storage refactoring verification (Section G1)."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.receiver_acquisition.runner import (
    run_acquisition_loop,
)
from latentmas_reprop.application.receiver_acquisition.sampling import SecretDigitSample
from latentmas_reprop.domain.models.tensor_artifact import ConfigurationMismatchError
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
    get_path_resolver,
)
from latentmas_reprop.infrastructure.storage import (
    ResearchDataReader,
    is_condition_complete,
    load_tensor,
    save_condition_result,
    save_tensor,
    verify_run_config,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker
from tools.migrations.storage_v1_to_v2.migrate import StorageMigrator


def test_distinct_run_ids_for_identical_config(tmp_path):
    """Two new runs of same config get distinct run_ids and directories."""
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)
    use_case = BenchmarkUseCase(
        dataset_port=MagicMock(load=MagicMock(return_value=[])),
        cache_port=cache,
        evaluator_port=MagicMock(),
    )
    args1 = SimpleNamespace(
        method="latent_mas",
        model_name="Qwen/test",
        task="gsm8k",
        prompt="sequential",
        seed=42,
        latent_steps=10,
        max_samples=0,
        generate_bs=1,
        max_new_tokens=64,
        temperature=0.6,
        top_p=0.95,
        split="test",
        tracking_experiment_name="test_exp",
    )
    args2 = SimpleNamespace(**vars(args1))
    model = SimpleNamespace(dtype_name="float16", load_time_sec=0.0)

    m1, _ = use_case.execute(model, args1)
    m2, _ = use_case.execute(model, args2)

    assert m1.run_id != m2.run_id
    assert resolver.get_experiment_dir(m1.run_id) != resolver.get_experiment_dir(
        m2.run_id
    )


def test_config_mismatch_raises_on_resume(tmp_path):
    """Missing or mismatched decisive config keys raise ConfigurationMismatchError."""
    exp_dir = tmp_path / "exp1"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": "test_model",
                "latent_steps": 10,
                "seed": 42,
                "split": "test",
                "sample_keys": ["k1", "k2"],
            }
        ),
        encoding="utf-8",
    )
    # Mismatched seed
    with pytest.raises(ConfigurationMismatchError):
        verify_run_config(
            exp_dir,
            {
                "model_name": "test_model",
                "latent_steps": 10,
                "seed": 999,
                "split": "test",
                "sample_keys": ["k1", "k2"],
            },
        )
    # Missing decisive key in existing
    with pytest.raises(ConfigurationMismatchError):
        verify_run_config(
            exp_dir,
            {
                "model_name": "test_model",
                "latent_steps": 10,
                "seed": 42,
                "split": "test",
                "prompt": "chain_of_thought",
                "sample_keys": ["k1", "k2"],
            },
        )


def test_incomplete_condition_triggers_reexecution(tmp_path):
    """Error result, missing files, or corrupted result.json marked incomplete."""
    cond_dir = tmp_path / "cond"
    cond_dir.mkdir()
    # 1. Error result
    (cond_dir / "result.json").write_text(
        json.dumps({"error": "OOM", "files": ["hidden.pt"]}), encoding="utf-8"
    )
    assert not is_condition_complete(cond_dir)

    # 2. Corrupt JSON
    (cond_dir / "result.json").write_text("{bad_json:", encoding="utf-8")
    assert not is_condition_complete(cond_dir)

    # 3. Missing declared file
    (cond_dir / "result.json").write_text(
        json.dumps({"files": ["hidden.pt"]}), encoding="utf-8"
    )
    assert not is_condition_complete(cond_dir)


def test_actual_kv_tensor_save_and_load_on_cpu(tmp_path):
    """Arbitrary KV cache tuple structure saved and loaded correctly on CPU."""
    kv_path = tmp_path / "shared" / "kv.pt"
    real_kv = (
        (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),
        (torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16)),
    )
    save_tensor(kv_path, real_kv)
    loaded_kv = load_tensor(kv_path)
    assert len(loaded_kv) == 2
    assert torch.equal(loaded_kv[0][0], real_kv[0][0])
    assert torch.equal(loaded_kv[1][1], real_kv[1][1])


def test_acquisition_lazy_cache_build():
    """Acquisition loop builds sender cache only when needed."""
    resolver = get_path_resolver()
    samples = [
        SecretDigitSample(sample_id="s0", sample_index=0, sample_key="s0", digit=1),
        SecretDigitSample(sample_id="s1", sample_index=1, sample_key="s1", digit=2),
    ]
    # Mark s0 complete on disk
    c_dir = resolver.get_condition_dir("run_acq", "s0", "full_own")
    save_condition_result(
        c_dir,
        {
            "sample_id": "s0",
            "condition": "own",
            "context_mode": "full",
            "target_digit": 1,
        },
    )

    args = SimpleNamespace(
        acquisition_conditions=["own"],
        context_modes=["full"],
        latent_steps=5,
        save_hidden_states=False,
    )
    model = MagicMock()
    model.device = torch.device("cpu")

    rec_mock = MagicMock()
    rec_mock.hidden_state_artifact_key = None
    rec_mock.to_dict.return_value = {
        "sample_id": "s1",
        "condition": "own",
        "context_mode": "full",
    }
    rec_mock.sample_index = 1
    rec_mock.predicted_digit = 2
    rec_mock.source_probability = 0.9
    rec_mock.content_follow_correct = True
    rec_mock.error = None

    with (
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.build_sender_cache"
        ) as mock_build,
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.forward_receiver_observation",
            return_value=(rec_mock, None),
        ),
    ):
        bundle_mock = MagicMock()
        bundle_mock.build_latency_sec = 0.05
        bundle_mock.full = None
        mock_build.return_value = bundle_mock

        records, _, _ = run_acquisition_loop(
            model=model,
            args=args,
            samples=samples,
            cross_indices=[0, 0],
            receiver_ids=torch.zeros((1, 2), dtype=torch.long),
            receiver_mask=torch.ones((1, 2), dtype=torch.long),
            mapping={},
            tracker_port=None,
            run_id="run_acq",
        )
        # s0 was complete, so build_sender_cache should only be called once for s1!
        assert mock_build.call_count == 1
        assert len(records) == 2


def test_mlflow_linking_and_failure_close(tmp_path):
    """MLflow records data_dir tag and ends run with FAILED on exception."""
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(path_resolver=resolver)
    run = tracker.start_run("test_exp", run_name="exp_link")
    run_id = run.info.run_id

    data_dir = resolver.get_experiment_dir("custom_storage_01")
    tracker.set_data_dir("custom_storage_01", data_dir)

    client = tracker.client
    r_info = client.get_run(run_id)
    assert r_info.data.tags.get("data_dir") == str(data_dir.resolve())
    assert r_info.data.tags.get("storage_run_id") == "custom_storage_01"

    tracker.end_run(status="FAILED")
    r_after = client.get_run(run_id)
    assert r_after.info.status == "FAILED"


def test_reader_path_traversal_blocked(tmp_path):
    """Path traversal characters are strictly rejected with ValueError."""
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(path_resolver=resolver)

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_artifact("run_1", "../secret.json")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="../bad", condition="own")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="s1", condition="../bad")


def test_migration_tool_experiment_artifact_location_and_sample_extraction(tmp_path):
    """Migration updates SQLite experiments.artifact_location and extracts samples.jsonl."""
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()

    # Create dummy mlflow.db with valid MLflow schema columns
    db_path = src / "mlflow" / "mlflow.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("CREATE TABLE experiments (experiment_id TEXT, artifact_location TEXT)")
    cur.execute(
        "INSERT INTO experiments VALUES ('1', 'file:///old/path/.cache/mlflow_artifacts/1')"
    )
    cur.execute(
        "CREATE TABLE runs (run_uuid TEXT, experiment_id TEXT, name TEXT, status TEXT, artifact_uri TEXT)"
    )
    cur.execute(
        "INSERT INTO runs VALUES ('r1', '1', 'run1', 'FINISHED', 'file:///old/path/.cache/mlflow_artifacts/1/r1/artifacts')"
    )
    conn.commit()
    conn.close()

    # Create dummy benchmark run under evaluation/runs/<run_id>/samples.jsonl
    run_dir = src / "evaluation" / "runs" / "bench_01"
    run_dir.mkdir(parents=True)
    (run_dir / "samples.jsonl").write_text(
        json.dumps({"sample_key": "k_123", "prediction": "4"}) + "\n",
        encoding="utf-8",
    )

    migrator = StorageMigrator(src, dest, dry_run=False)
    migrator.run()

    # Verify SQLite experiments table was updated
    dest_db = dest / "mlflow" / "mlflow.db"
    conn2 = sqlite3.connect(str(dest_db))
    cur2 = conn2.cursor()
    cur2.execute("SELECT artifact_location FROM experiments WHERE experiment_id='1'")
    loc = cur2.fetchone()[0]
    assert "mlflow_artifacts" in loc
    assert ".cache" not in loc
    conn2.close()

    # Verify sample benchmark directory extracted
    sample_res = (
        dest
        / "experiments"
        / "bench_01"
        / "samples"
        / "k_123"
        / "benchmark"
        / "result.json"
    )
    assert sample_res.is_file()
    data = json.loads(sample_res.read_text(encoding="utf-8"))
    assert data["sample_key"] == "k_123"
