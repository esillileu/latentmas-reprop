"""Exercise actual MLflow export for every receiver cut and execution failures."""

from types import SimpleNamespace

import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_acquisition import trajectory
from latentmas_reprop.application.receiver_acquisition.sampling import (
    generate_secret_digit_samples,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.tracking import MLflowTracker


@pytest.mark.parametrize("fail_step", [None, 4])
def test_sample_trace_exports_all_steps_or_records_forward_failure(
    tmp_path, monkeypatch, fail_step
):
    sample = generate_secret_digit_samples(100, 42)[0]
    full = ((torch.zeros(1, 1, 23, 1), torch.ones(1, 1, 23, 1)),)
    builds = []

    def build(model, args, target):
        builds.append(args.latent_steps)
        return SimpleNamespace(
            full=full, prompt_len=3, full_len=23, build_latency_sec=0.01
        )

    def forward(*args, **kwargs):
        if fail_step and kwargs["position_start"] == 3 + fail_step:
            raise RuntimeError("receiver failed")
        return SimpleNamespace(
            logits=torch.arange(32).float()[None], hidden_states=None
        )

    model = SimpleNamespace(
        model_name="Qwen/Qwen3-0.6B",
        device="cpu",
        forward_next_token_batch=forward,
        tokenizer=SimpleNamespace(convert_ids_to_tokens=str, decode=str),
    )
    receiver = (
        torch.tensor([[1, 2]]),
        torch.ones(1, 2),
        {"candidates": {str(d): {"contextual_token_id": d} for d in range(10)}},
    )
    monkeypatch.setattr(trajectory, "build_sender_cache", build)
    previous_uri = mlflow.get_tracking_uri()
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'traces.db'}",
        artifact_location=str(tmp_path / "artifacts"),
        path_resolver=PathResolver(tmp_path),
    )
    run = tracker.start_run("trajectory-tracing", run_name="sample")
    try:
        if fail_step:
            with pytest.raises(RuntimeError, match="receiver failed"):
                trajectory.collect_trajectory(
                    model, [sample], receiver, tmp_path / "records.jsonl", tracker
                )
        else:
            rows = trajectory.collect_trajectory(
                model, [sample], receiver, tmp_path / "records.jsonl", tracker
            )
            untraced = trajectory.collect_trajectory(
                model, [sample], receiver, tmp_path / "untraced.jsonl"
            )
            for traced, plain in zip(rows, untraced, strict=True):
                assert {k: v for k, v in traced.items() if k != "latency_sec"} == {
                    k: v for k, v in plain.items() if k != "latency_sec"
                }
        tracker.flush_traces()
        client = MlflowClient(tracking_uri=tracker.tracking_uri)
        traces = client.search_traces(
            locations=[run.info.experiment_id], run_id=run.info.run_id
        )
        assert len(traces) == 1
        spans = traces[0].data.spans
        root = next(s for s in spans if s.parent_id is None)
        assert root.name == "secret_digit_sample"
        assert traces[0].info.tags["sample_id"] == sample.sample_id
        sender = next(s for s in spans if s.name == "build_sender_context")
        assert sender.parent_id == root.span_id
        receivers = [s for s in spans if s.span_type == "LLM"]
        assert all(s.parent_id == root.span_id for s in receivers)
        if fail_step:
            assert root.status.status_code.name == "ERROR"
            assert len(receivers) == 1 + fail_step
        else:
            assert len(receivers) == 21
            own = [s for s in receivers if s.inputs["condition"] == "own"]
            assert {s.inputs["latent_steps"] for s in own} == set(range(1, 21))
            assert all(
                s.outputs["receiver_position_start"] == 3 + s.inputs["latent_steps"]
                for s in own
            )
            assert len(root.outputs["cells"]) == 21
            names = {a.name for a in traces[0].info.assessments}
            assert {"expected_target_digit", "execution_success"} <= names
            assert all(
                f"latent_only_own_step_{k}_source_follow" in names for k in range(1, 21)
            )
            assert builds == [20, 20]
    finally:
        tracker.flush_traces()
        tracker.end_run()
        mlflow.set_tracking_uri(previous_uri)
