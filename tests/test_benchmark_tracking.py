"""Tests for benchmark MLflow tracking, experiment enforcement, and tracing."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.domain.models import BenchmarkMetrics
from latentmas_reprop.domain.ports.tracking_port import ExperimentTrackerPort
from run.cli import parse_args
from run.runner import run_benchmark


class DummyTracker(ExperimentTrackerPort):
    def __init__(self):
        self.runs = []
        self.params = []
        self.metrics = []
        self.dicts = []
        self.artifacts = []
        self.ended_status = None
        self.traces = []
        self.spans = []
        self.expectations = []
        self.feedbacks = []
        self.flushed = False

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict | None = None,
    ):
        self.runs.append(
            {"experiment_name": experiment_name, "run_name": run_name, "tags": tags}
        )

    def log_params(self, params: dict):
        self.params.append(params)

    def log_metrics(self, metrics: dict, step: int | None = None):
        self.metrics.append(metrics)

    def log_artifact(self, local_path, artifact_path: str | None = None):
        self.artifacts.append((str(local_path), artifact_path))

    def log_dict(self, dictionary: dict, artifact_file: str):
        self.dicts.append((dictionary, artifact_file))

    def end_run(self, status: str = "FINISHED"):
        self.ended_status = status

    def start_sample_trace(
        self,
        name: str,
        inputs: dict,
        tags: dict | None = None,
        request_preview: str | None = None,
    ):
        trace_data = {
            "name": name,
            "inputs": inputs,
            "tags": tags,
            "trace_id": "trace-123",
        }
        self.traces.append(trace_data)

        class TraceContext:
            def __init__(self, parent):
                self.parent = parent
                self.trace_id = "trace-123"

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def set_outputs(self, outputs):
                trace_data["outputs"] = outputs

        return TraceContext(self)

    def start_span(
        self, name: str, span_type: str = "UNKNOWN", inputs: dict | None = None
    ):
        span_data = {"name": name, "span_type": span_type, "inputs": inputs}
        self.spans.append(span_data)

        class SpanContext:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def set_outputs(self, outputs):
                span_data["outputs"] = outputs

        return SpanContext()

    def log_expectation(
        self, trace_id: str, name: str, value, source_id: str = "ground_truth"
    ):
        self.expectations.append({"trace_id": trace_id, "name": name, "value": value})

    def log_feedback(
        self,
        trace_id: str,
        name: str,
        value,
        source_id: str = "evaluator",
        rationale: str | None = None,
    ):
        self.feedbacks.append(
            {"trace_id": trace_id, "name": name, "value": value, "rationale": rationale}
        )

    def flush_traces(self):
        self.flushed = True


def test_benchmark_use_case_requires_tracking_experiment_name():
    use_case = BenchmarkUseCase()
    args = SimpleNamespace(
        method="baseline",
        model_name="test_model",
        task="gsm8k",
        split="test",
        seed=42,
        max_samples=1,
    )
    model = MagicMock()
    with pytest.raises(
        ValueError, match="MLflow tracking experiment name must be specified"
    ):
        use_case.execute(model, args)


def test_run_benchmark_requires_tracking_experiment_name():
    args = SimpleNamespace(
        method="baseline",
        model_name="test_model",
        task="gsm8k",
        seed=42,
    )
    with pytest.raises(
        ValueError, match="MLflow tracking experiment name must be specified"
    ):
        run_benchmark(args)


def test_reprop_configs_resolve_to_latentmas_reprop_experiment():
    bs_args = parse_args(
        [
            "-c",
            "configs/lmas/reprop/bs_q38_gsm8k.yaml",
            "--method",
            "baseline",
        ]
    )
    assert bs_args.tracking_experiment_name == "latentmas-reprop"

    lm_args = parse_args(
        [
            "-c",
            "configs/lmas/reprop/lm_q38_gsm8k.yaml",
            "--latent_steps",
            "10",
        ]
    )
    assert lm_args.tracking_experiment_name == "latentmas-reprop"


def test_benchmark_traces_and_spans_for_textmas(tmp_path):
    tracker = DummyTracker()
    dataset_port = MagicMock()
    dataset_port.load.return_value = [
        {"question": "What is 2+2?", "gold": "4", "solution": "2+2=4"}
    ]
    cache_port = MagicMock()
    cache_port.get_layer_path.return_value = tmp_path

    use_case = BenchmarkUseCase(
        dataset_port=dataset_port, cache_port=cache_port, tracker_port=tracker
    )

    args = SimpleNamespace(
        method="text_mas",
        model_name="mock_model",
        task="gsm8k",
        split="test",
        prompt="sequential",
        seed=42,
        max_samples=1,
        max_new_tokens=64,
        temperature=0.7,
        top_p=0.95,
        generate_bs=1,
        tracking_experiment_name="latentmas-reprop",
    )

    # Mock method execution result
    mock_method = MagicMock()
    mock_method.run_batch.return_value = [
        {
            "question": "What is 2+2?",
            "gold": "4",
            "solution": "2+2=4",
            "prediction": "4",
            "raw_prediction": "4",
            "agents": [
                {
                    "name": "Planner",
                    "role": "planner",
                    "input": "Plan 2+2",
                    "output": "Calculate 2+2",
                },
                {
                    "name": "Critic",
                    "role": "critic",
                    "input": "Review 2+2",
                    "output": "Looks fine",
                },
                {
                    "name": "Judger",
                    "role": "judger",
                    "input": "Final answer for 2+2",
                    "output": "The answer is 4",
                },
            ],
            "correct": True,
        }
    ]

    use_case.create_method = MagicMock(return_value=mock_method)
    model = MagicMock(dtype_name="float16", load_time_sec=0.1)

    metrics, preds = use_case.execute(model, args)

    assert isinstance(metrics, BenchmarkMetrics)
    assert len(preds) == 1
    assert tracker.runs[0]["experiment_name"] == "latentmas-reprop"
    assert tracker.ended_status == "FINISHED"
    assert tracker.flushed is True

    # Check trace and spans
    assert len(tracker.traces) == 1
    assert tracker.traces[0]["name"] == "sample_1"
    assert tracker.traces[0]["inputs"]["question"] == "What is 2+2?"

    # Check that each agent step was recorded as a span
    span_names = [s["name"] for s in tracker.spans]
    assert "Planner_planner" in span_names
    assert "Critic_critic" in span_names
    assert "Judger_judger" in span_names

    planner_span = next(s for s in tracker.spans if s["name"] == "Planner_planner")
    assert planner_span["inputs"]["prompt"] == "Plan 2+2"
    judger_span = next(s for s in tracker.spans if s["name"] == "Judger_judger")
    assert judger_span["inputs"]["prompt"] == "Final answer for 2+2"

    # Check expectation and feedback
    assert len(tracker.expectations) == 1
    assert tracker.expectations[0]["value"] == "4"
    assert len(tracker.feedbacks) == 1
    assert tracker.feedbacks[0]["value"] is True
