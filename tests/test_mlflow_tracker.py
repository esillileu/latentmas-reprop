import os
from pathlib import Path

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def test_mlflow_tracker_local_default(tmp_path: Path):
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(path_resolver=resolver)

    assert not tracker.is_remote
    assert tracker.tracking_uri.startswith("sqlite:///")
    assert str(resolver.cache_dir / "mlflow.db") in tracker.tracking_uri
    assert tracker.artifact_location is not None
    assert str(resolver.cache_dir / "mlflow_artifacts") in tracker.artifact_location
    assert Path(tracker.artifact_location).is_dir()


def test_mlflow_tracker_remote_env_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://remote-server:5000")
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(path_resolver=resolver)

    assert tracker.is_remote
    assert tracker.tracking_uri == "http://remote-server:5000"
    # When remote, do not override local artifact location
    assert tracker.artifact_location is None


def test_mlflow_tracker_end_to_end_local(tmp_path: Path):
    resolver = PathResolver(tmp_path)
    test_db = tmp_path / "test_run.db"
    test_artifacts = tmp_path / "artifacts"
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{test_db}",
        artifact_location=str(test_artifacts),
        path_resolver=resolver,
    )

    tracker.start_run(experiment_name="test_experiment", run_name="test_run")
    tracker.log_params({"param_a": 123, "param_b": "value"})
    tracker.log_metrics({"acc": 0.95, "loss": 0.05})

    artifact_file = tmp_path / "sample.txt"
    artifact_file.write_text("dummy artifact")
    tracker.log_artifact(artifact_file)
    tracker.log_dict({"status": "ok"}, "status.json")

    tracker.end_run()
    assert os.path.exists(test_db)
