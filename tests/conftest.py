import copy
import os
import shutil
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient


@pytest.fixture(scope="session")
def mlflow_sqlite_template(tmp_path_factory):
    """Create a pre-migrated MLflow SQLite DB once per test session to avoid Alembic migration overhead."""
    template_dir = tmp_path_factory.mktemp("mlflow_template")
    template_db = template_dir / "template.db"
    client = MlflowClient(f"sqlite:///{template_db}")
    exp_id = client.create_experiment("init")
    client.delete_experiment(exp_id)
    return template_db


@pytest.fixture
def make_mlflow_db(mlflow_sqlite_template):
    """Helper to quickly copy pre-migrated SQLite DB to a target path."""
    def _copy(dest_path: Path) -> str:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(mlflow_sqlite_template, dest_path)
        return f"sqlite:///{dest_path}"
    return _copy


@pytest.fixture(autouse=True)
def isolate_test_environment(monkeypatch, tmp_path_factory, mlflow_sqlite_template):
    """Ensure tests run strictly isolated from ambient MLflow servers and production runs."""
    test_env_dir = tmp_path_factory.mktemp("test_env")
    storage_dir = test_env_dir / "storage"
    cache_dir = test_env_dir / ".cache"
    mlflow_dir = storage_dir / "mlflow"
    artifacts_dir = test_env_dir / "artifacts"

    mlflow_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mlflow_sqlite_template, mlflow_dir / "mlflow.db")

    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(storage_dir))
    monkeypatch.setenv("LATENTMAS_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("LATENTMAS_MLFLOW_DIR", str(mlflow_dir))
    monkeypatch.setenv("LATENTMAS_ARTIFACTS_DIR", str(artifacts_dir))

    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.setenv("LATENTMAS_PROJECT_ROOT", str(repo_root))

    if "MLFLOW_TRACKING_URI" in os.environ:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


@pytest.fixture
def in_memory_tracker():
    """Provide a fast, in-memory ExperimentTrackerPort for unit tests."""
    from latentmas_reprop.infrastructure.tracking import InMemoryTracker

    return InMemoryTracker()


@pytest.fixture
def in_memory_storage():
    """Provide a fast, in-memory Storage service for unit tests."""
    from latentmas_reprop.infrastructure.storage import InMemoryStorage

    return InMemoryStorage()


@pytest.fixture
def preset_null_distribution():
    """Deterministic pre-computed null distribution for testing statistical probe significance tests."""
    return {
        "null_oof_accuracies": [0.1, 0.12, 0.08, 0.11, 0.09],
        "null_mean": 0.10,
        "null_std": 0.015,
        "fwer_p_value": 0.01,
        "significance": True,
    }


@pytest.fixture
def pilot():
    config = {
        "model_name": "Qwen/test",
        "task": "gsm8k",
        "split": "test",
        "prompt": "sequential",
        "seed": 42,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 2048,
        "answer_only_max_new_tokens": 64,
        "upstream_steps": [10, 40],
    }
    patterns = [
        (False, True, True, True),
        (True, True, False, True),
        (False, False, False, True),
        (False, True, False, False),
    ]
    records = []
    for i, pattern in enumerate(patterns):
        for (step, mode), correct in zip(
            [(10, "answer_only"), (10, "free"), (40, "answer_only"), (40, "free")],
            pattern,
            strict=True,
        ):
            records.append(
                {
                    "sample_id": str(i),
                    "sample_index": i,
                    "question": f"q{i}",
                    "gold": "1",
                    "upstream_latent_steps": step,
                    "receiver_mode": mode,
                    "prediction": "1" if correct else "0",
                    "raw_receiver_output": "answer",
                    "correct": correct,
                    "receiver_generated_tokens": 5,
                    "receiver_token_limit_reached": False,
                    "context_id": f"{i}:{step}",
                    "upstream_context_sequence_length": 100 + step,
                    "model": config["model_name"],
                    "seed": 42,
                    "config": copy.deepcopy(config),
                }
            )
    metadata = {
        "run_id": "run",
        "experiment_id": "1",
        "status": "FINISHED",
        "start_time": 0,
        "end_time": 1,
        "artifact_uri": "uri",
        "params": {k: str(v) for k, v in config.items()},
        "metrics": {},
        "tags": {},
    }
    return records, config, metadata
