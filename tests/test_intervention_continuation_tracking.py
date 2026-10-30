"""Regression coverage for MLflow reconciliation of saved intervention results."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def setup_run(tmp_path):
    wrapper = _build_mock_wrapper()
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=2,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_samples=1,
        no_progress=True,
        intervention_conditions=["rqle"],
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q1", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}", path_resolver=resolver
    )
    case = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=FileStorage(resolver),
        tracker_port=tracker,
    )
    return case, wrapper, args, tracker


def traces(tracker):
    experiment = tracker.client.get_experiment_by_name(
        "latentmas/intervention/continuation"
    )
    return list(
        tracker.client.search_traces(locations=[experiment.experiment_id], flush=True)
    )


def test_incremental_trace_reuse_and_backfill(setup_run):
    case, wrapper, args, tracker = setup_run
    case.execute(wrapper, args)
    first = traces(tracker)
    assert len(first) == 1
    old_id = first[0].info.trace_id
    run_id = tracker.client.search_runs([first[0].info.experiment_id])[0].info.run_id
    import json
    from pathlib import Path

    import torch

    from latentmas_reprop.infrastructure.storage.reader import ResearchDataReader

    run_dir = Path(tracker.client.get_run(run_id).data.tags["data_dir"])
    config = json.loads((run_dir / "config.json").read_text())
    assert config["step"] == 2 and len(config["result_paths"]) == 1
    reader = ResearchDataReader(
        mlflow_client=tracker.client,
        tensor_store=case.storage,
        path_resolver=tracker.resolver,
    )
    kv, hidden, _ = reader.load_intervention_state(run_id, "000000")
    state = case.storage.base_dir / config["state_path"]
    expected_kv, expected_hidden, _ = case.storage.assemble_shared_state(
        state, "000000", 2
    )
    assert torch.equal(hidden, expected_hidden)
    assert torch.equal(kv[0][0], expected_kv[0][0])
    path = Path(config["result_paths"][0])
    result_hidden, record = reader.load_intervention_result(
        run_id, "000000", 2, path.parts[-2], path.parts[-1]
    )
    direct_hidden, direct_record = case.storage.load_intervention_result(state / path)
    assert torch.equal(result_hidden, direct_hidden) and record == direct_record
    wrapper.model.forward.reset_mock()
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
    assert [t.info.trace_id for t in traces(tracker)] == [old_id]
    run = tracker.client.get_run(run_id)
    tracker.start_run(
        "latentmas/intervention/continuation",
        run_name=run.data.tags["rel_dir"],
        tags=run.data.tags,
    )
    with tracker.start_sample_trace(
        "unrelated", {}, tags={"sample_key": "000000", "comparison_id": "other"}
    ) as span:
        unrelated_id = span.trace_id
    tracker.flush_traces()
    tracker.end_run()
    args.intervention_conditions = ["rqlqe", "rqlrqe"]
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 2
    config = json.loads((run_dir / "config.json").read_text())
    assert len(config["result_paths"]) == 3
    all_traces = traces(tracker)
    assert unrelated_id in {trace.info.trace_id for trace in all_traces}
    updated = [trace for trace in all_traces if trace.info.trace_id != unrelated_id]
    assert len(updated) == 1 and updated[0].info.trace_id != old_id
    assert {s.name for s in updated[0].data.spans} >= {
        "common_latent",
        "decode_RQLE",
        "decode_RQLQE",
        "decode_RQLRQE",
    }
    assert (
        tracker.client.search_runs([updated[0].info.experiment_id])[0].info.run_id
        == run_id
    )
    tracker.client.delete_traces(
        experiment_id=updated[0].info.experiment_id,
        trace_ids=[updated[0].info.trace_id],
    )
    wrapper.model.forward.reset_mock()
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
    assert len(traces(tracker)) == 2
    assert unrelated_id in {trace.info.trace_id for trace in traces(tracker)}


def test_trace_failures_preserve_previous_result(setup_run, monkeypatch):
    from mlflow.tracking import MlflowClient

    from latentmas_reprop.application.intervention_continuation import tracking

    case, wrapper, args, tracker = setup_run
    case.execute(wrapper, args)
    first = traces(tracker)[0]
    old_id = first.info.trace_id
    args.intervention_conditions = ["rqlqe"]
    case.evaluator_port.evaluate.side_effect = RuntimeError("evaluation failed")
    with pytest.raises(RuntimeError, match="evaluation failed"):
        case.execute(wrapper, args)
    assert [t.info.trace_id for t in traces(tracker)] == [old_id]
    case.evaluator_port.evaluate.side_effect = None
    from latentmas_reprop.infrastructure.storage.tensor_store import (
        load_json,
        save_json_atomic,
    )

    record_path = next(
        case.storage.base_dir.glob(
            "intervention_continuation/*/*/*/samples/000000/results/ls2/*/ple/result.json"
        )
    )
    saved = load_json(record_path)
    saved["trace_id"] = old_id
    save_json_atomic(record_path, saved)
    args.intervention_conditions = ["rqlqe"]
    original = tracking.trace_intervention_continuation_sample
    monkeypatch.setattr(
        tracking, "trace_intervention_continuation_sample", lambda *a, **kw: None
    )
    with pytest.raises(RuntimeError, match="Trace creation failed"):
        case.execute(wrapper, args)
    assert [t.info.trace_id for t in traces(tracker)] == [old_id]
    monkeypatch.setattr(tracking, "trace_intervention_continuation_sample", original)
    verify = tracking._verified
    monkeypatch.setattr(tracking, "_verified", lambda *a: False)
    with pytest.raises(RuntimeError, match="Trace verification failed"):
        case.execute(wrapper, args)
    assert old_id in {t.info.trace_id for t in traces(tracker)}
    monkeypatch.setattr(tracking, "_verified", verify)
    delete = MlflowClient.delete_traces

    def fail_delete(*args, **kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(MlflowClient, "delete_traces", fail_delete)
    wrapper.model.forward.reset_mock()
    with pytest.raises(RuntimeError, match="delete failed"):
        case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
    assert len(traces(tracker)) == 2
    assert old_id in {t.info.trace_id for t in traces(tracker)}
    monkeypatch.setattr(MlflowClient, "delete_traces", delete)
    case.execute(wrapper, args)
    assert len(traces(tracker)) == 1
    assert "trace_id" not in load_json(record_path)
    for summary in case.storage.base_dir.glob(
        "intervention_continuation/*/*/*/runs/*/samples.jsonl"
    ):
        assert old_id not in summary.read_text()


def test_different_steps_and_p1_use_different_runs(setup_run):
    case, wrapper, args, tracker = setup_run
    case.execute(wrapper, args)
    first_id = traces(tracker)[0].info.trace_id
    case.dataset_port.load.return_value.append({"question": "Q2", "gold": "1"})
    args.max_samples = 2
    case.execute(wrapper, args)
    assert len(traces(tracker)) == 2
    assert first_id in {t.info.trace_id for t in traces(tracker)}
    args.max_samples = 1
    args.latent_steps = 3
    case.execute(wrapper, args)
    args.p1_variant = "question_only"
    args.intervention_conditions = ["qle"]
    case.execute(wrapper, args)
    found = traces(tracker)
    assert len(found) == 4
    experiment = tracker.client.get_experiment_by_name(
        "latentmas/intervention/continuation"
    )
    assert len(tracker.client.search_runs([experiment.experiment_id])) == 3
