"""Receiver collection reuses paired contexts and baseline trajectories."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from latentmas_reprop.domain.ports.tracking_port import DummySpan, ExperimentTrackerPort
from src.run.cli import parse_run_matrix


@pytest.mark.parametrize("persist_traces", [False, True])
def test_collection_builds_each_context_once_and_uses_other_sample(
    monkeypatch, tmp_path, persist_traces
):
    import latentmas_reprop.application.receiver_compute_preflight.trajectory as trajectory
    import latentmas_reprop.application.receiver_compute_preflight_use_case as module

    class Tracker(ExperimentTrackerPort):
        def log_dict(self, *args, **kwargs):
            pass

        def __init__(self):
            self.artifacts = {}
            self.status = None
            self.roots = []
            self.spans = []
            self.feedback = []
            self.current_trace = None

        @property
        def active_run_id(self):
            return backend.active_run_id if persist_traces else "test-run"

        @contextmanager
        def start_sample_trace(self, name, inputs, **kwargs):
            trace_id = f"trace-{len(self.roots)}"
            self.current_trace = trace_id
            self.roots.append((name, inputs, kwargs))
            span = SimpleNamespace(
                trace_id=trace_id,
                set_outputs=lambda value: None,
                set_attribute=lambda *args: None,
                set_status=lambda *args: None,
            )
            try:
                yield span
            finally:
                self.current_trace = None

        @contextmanager
        def start_span(self, name, span_type, inputs=None):
            assert self.current_trace is not None
            self.spans.append((self.current_trace, name, span_type, inputs))

            class Span(DummySpan):
                @property
                def trace_id(span):
                    return self.current_trace

            yield Span()

        def log_feedback(self, trace_id, name, value, **kwargs):
            self.feedback.append((trace_id, name, value))

        def start_run(self, **kwargs):
            self.run_options = kwargs

        def log_params(self, config):
            self.config = config

        def log_artifact(self, path, **kwargs):
            self.artifacts[path.name] = path.read_text()

        def log_metrics(self, metrics):
            pass

        def flush_traces(self):
            pass

        def end_run(self, status):
            self.status = status

    class Dataset:
        def load(self, **kwargs):
            return [
                {"question": "first", "gold": "1"},
                {"question": "second", "gold": "2"},
            ]

    tracker = Tracker()
    if persist_traces:
        from latentmas_reprop.infrastructure.tracking.mlflow_tracker import (
            MLflowTracker,
        )

        backend = MLflowTracker(
            tracking_uri=f"sqlite:///{tmp_path / 'traces.db'}",
            artifact_location=str(tmp_path / "artifacts"),
        )
        # Exercise the real adapter with the same paired collection and fake inference.
        for method_name in (
            "start_run",
            "log_params",
            "log_artifact",
            "log_metrics",
            "flush_traces",
            "end_run",
            "log_feedback",
        ):
            original = getattr(tracker, method_name)
            remote = getattr(backend, method_name)

            def both(*args, _original=original, _remote=remote, **kwargs):
                _original(*args, **kwargs)
                return _remote(*args, **kwargs)

            setattr(tracker, method_name, both)
        tracker.start_sample_trace = backend.start_sample_trace
        tracker.start_span = backend.start_span
        tracker.update_current_trace = backend.update_current_trace
        tracker.log_expectation = backend.log_expectation
    args = parse_run_matrix(
        [
            "-c",
            "lmas/receiver_compute_preflight/gsm8k",
            "--max_samples",
            "2",
            "--receiver_budgets",
            "2,4,free",
            "--max_new_tokens",
            "8",
            "--bootstrap_count",
            "10",
        ]
    )[0]
    calls, generation_calls = [], []

    def build(method, item, step, width, context_id, tracker, runtime):
        assert width is None
        calls.append((item["question"], step))
        return (
            None,
            [[{"latent_steps": step}]],
            {"handoff_positions": len(item["question"])},
        )

    def generate(method, item, context, limit, budgets=(), report_progress=False):
        generation_calls.append((item["question"], method.latent_steps, limit))
        return [1, 2][:limit], {
            "receiver_input_ids": [5],
            "receiver_prompt": "prompt",
            "receiver_thinking_open": False,
            "receiver_cache_positions": 0,
            "receiver_prompt_tokens": 1,
            "receiver_latency_sec": 0.1,
            "receiver_budget_latency_sec": {str(b): 0.05 for b in budgets},
        }

    monkeypatch.setattr(module, "build_upstream", build)
    monkeypatch.setattr(trajectory, "generate_receiver", generate)
    model = SimpleNamespace(
        device="cpu",
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids[-1])),
    )
    metrics, rows = module.ReceiverComputePreflightUseCase(
        dataset_port=Dataset(), tracker_port=tracker
    ).execute(model, args)
    assert len(calls) == 4
    assert len(generation_calls) == 10
    assert sum(u == 0 for _, u, _ in generation_calls) == 2
    assert len(rows) == 36
    assert metrics["sample_count"] == 2
    assert tracker.status == "FINISHED"
    assert "sample_results.jsonl" in tracker.artifacts
    assert all(
        r["sample_id"] != r["donor_id"]
        for r in rows
        if r["handoff_condition"] == "mismatched"
    )
    assert all(r["valid"] for r in rows)

    mismatched = [r for r in rows if r["handoff_condition"] == "mismatched"]
    assert all(r["donor_length_abs_delta"] == 1 for r in mismatched)
    assert {r["donor_length_delta"] for r in mismatched} == {-1, 1}
    assert all(
        r["mean_receiver_latency_sec"] == 0.05
        for r in metrics["curves"]
        if r["receiver_budget"] == 2
    )

    import json

    assert "Qwen3" in tracker.run_options["run_name"]
    assert "smoke" in tracker.run_options["run_name"]
    assert "_u10-20_r2-4-free_seed" in tracker.run_options["run_name"]
    if not persist_traces:
        assert len(tracker.roots) == 10  # two baseline, four upstream, four paired
        assert len([s for s in tracker.spans if s[2] == "LLM"]) == 10
        assert len([s for s in tracker.spans if s[2] == "EVALUATOR"]) == 36
    assert all(r["trace_id"] and r["receiver_trace_id"] for r in rows)
    assert all(r["upstream_trace_id"] for r in rows if r["donor_id"])
    manifest = json.loads(tracker.artifacts["trace_manifest.json"])
    assert len(manifest) == 10
    assert all(m["execution_success"] for m in manifest)
    assert len([f for f in tracker.feedback if f[1].endswith("_correct")]) == 36

    if persist_traces:
        import mlflow

        for entry in manifest:
            trace = mlflow.get_trace(entry["trace_id"])
            assert trace is not None
            names = {span.name for span in trace.data.spans}
            # The upstream builder is mocked in this test.
            assert (
                set(entry["required_span_names"])
                - {"build_upstream_10", "build_upstream_20"}
                <= names
            )
            assessments = {a.name for a in trace.info.assessments}
            assert set(entry["required_assessments"]) <= assessments
