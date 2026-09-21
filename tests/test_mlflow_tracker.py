import os
from pathlib import Path

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def test_mlflow_tracker_local_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
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


def test_mlflow_tracker_tracing_and_assessments(tmp_path: Path):
    import mlflow

    resolver = PathResolver(tmp_path)
    test_db = tmp_path / "test_traces.db"
    test_artifacts = tmp_path / "trace_artifacts"
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{test_db}",
        artifact_location=str(test_artifacts),
        path_resolver=resolver,
    )

    tracker.start_run(experiment_name="test_traces_exp", run_name="pilot_test")

    with tracker.start_sample_trace(
        name="latentmas_sample",
        inputs={"sample_id": "sample_0", "question": "1+1=?", "gold": "2"},
        tags={"task": "gsm8k", "model": "test_model", "pilot": "true"},
        request_preview="[sample_0] 1+1=?",
    ) as root:
        with tracker.start_span(
            name="build_own_context",
            span_type="CHAIN",
            inputs={"sample_id": "sample_0"},
        ) as b_span:
            b_span.set_outputs({"cache_present": True, "build_latency_sec": 0.05})

        with tracker.start_span(
            name="decode_own",
            span_type="LLM",
            inputs={"condition": "own", "sample_id": "sample_0"},
        ) as d_span:
            d_span.set_token_usage(prompt_tokens=10, completion_tokens=5)
            d_span.set_outputs({"prediction": "2", "correct": True})

        root.set_outputs({"own_prediction": "2", "own_correct": True})
        tracker.update_current_trace(response_preview="own: 2 (PASS)")
        trace_id = root.trace_id

    assert trace_id is not None
    tracker.log_expectation(
        trace_id=trace_id, name="expected_answer", value="2", source_id="ground_truth"
    )
    tracker.log_feedback(
        trace_id=trace_id, name="own_correct", value=True, source_id="evaluator"
    )
    tracker.flush_traces()
    tracker.end_run()

    # Verify directly via MLflow query API
    trace = mlflow.get_trace(trace_id)
    assert trace is not None
    span_names = [s.name for s in trace.data.spans]
    assert "latentmas_sample" in span_names
    assert "build_own_context" in span_names
    assert "decode_own" in span_names

    # Verify assessments recorded
    assessments = trace.info.assessments or []
    assert len(assessments) >= 2
    assessment_names = [a.name for a in assessments]
    assert "expected_answer" in assessment_names
    assert "own_correct" in assessment_names
