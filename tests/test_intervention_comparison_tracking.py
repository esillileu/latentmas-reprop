"""Persisted MLflow traces for comparisons and inference-free trace recovery."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from mlflow.tracking import MlflowClient
from test_intervention_comparison import comparison_case

from latentmas_reprop.application.intervention_continuation.tracking_verification import (
    verify_trace_spans,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import ResearchDataReader
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def test_comparison_trace_has_four_conditions_and_recovers_without_inference(
    tmp_path, monkeypatch, make_mlflow_db
):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(
        tracking_uri=make_mlflow_db(tmp_path / "mlflow.db"),
        path_resolver=resolver,
    )
    case, wrapper, args = comparison_case(tmp_path, tracker)

    def traces():
        experiment = tracker.client.get_experiment_by_name(
            "latentmas/intervention/continuation"
        )
        return list(
            tracker.client.search_traces(
                locations=[experiment.experiment_id], flush=True
            )
        )

    summary, _ = case.execute(wrapper, args)
    first = traces()
    assert len(first) == 1
    trace = first[0]
    assert (
        len(trace.data.spans) == 12
    )  # root, initial states, four decodes, six role prefixes
    assert {span.name for span in trace.data.spans} >= {
        "initial_states",
        "decode_BASELINE",
        "decode_LATENT_ONLY",
        "decode_INTERVENTION_ONLY",
        "decode_LATENT_INTERVENTION",
    }
    assert trace.info.tags["recording"] == "saved_results"
    incomplete = SimpleNamespace(
        info=trace.info,
        data=SimpleNamespace(
            spans=[span for span in trace.data.spans if span.name != "critic"]
        ),
    )
    assert not verify_trace_spans(
        incomplete,
        "000000",
        trace.info.tags["comparison_id"],
        trace.info.tags["result_fingerprint"],
        {condition.upper() for condition in args.intervention_conditions},
    )
    spans = {span.name: span for span in trace.data.spans}
    assert set(spans["initial_states"].inputs["initial_states"]) == {
        "planner",
        "single_agent",
    }
    assert spans["decode_LATENT_INTERVENTION"].inputs["latent_steps_executed"] == 3
    experiment_id = trace.info.experiment_id
    runs = tracker.client.search_runs([experiment_id])
    assert len(runs) == 1
    run_id = runs[0].info.run_id
    reader = ResearchDataReader(
        mlflow_client=tracker.client, tensor_store=case.storage, path_resolver=resolver
    )
    _, _, meta = reader.load_intervention_state(
        run_id, "000000", condition="latent_only"
    )
    assert meta["latent_steps"] == 2
    assert Path(runs[0].data.tags["data_dir"]).is_dir()
    wrapper.model.forward.reset_mock()
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
    assert traces()[0].info.trace_id == trace.info.trace_id
    tracker.client.delete_traces(
        experiment_id=experiment_id, trace_ids=[trace.info.trace_id]
    )
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
    assert len(traces()) == 1
    assert traces()[0].info.trace_id != trace.info.trace_id
    assert summary["sample_count"] == 1


def test_adding_comparison_conditions_replaces_only_its_sample_trace(
    tmp_path, monkeypatch, make_mlflow_db
):
    monkeypatch.setenv("MLFLOW_ENABLE_ASYNC_TRACE_LOGGING", "false")
    tracker = MLflowTracker(
        tracking_uri=make_mlflow_db(tmp_path / "mlflow.db"),
        path_resolver=PathResolver(tmp_path),
    )
    case, wrapper, args = comparison_case(tmp_path, tracker)
    args.intervention_conditions = ["baseline"]
    case.execute(wrapper, args)
    experiment = tracker.client.get_experiment_by_name(
        "latentmas/intervention/continuation"
    )
    old = next(
        iter(
            tracker.client.search_traces(
                locations=[experiment.experiment_id], flush=True
            )
        )
    )
    args.intervention_conditions = ["intervention_only"]
    case.execute(wrapper, args)
    traces = list(
        tracker.client.search_traces(locations=[experiment.experiment_id], flush=True)
    )
    assert len(traces) == 1 and traces[0].info.trace_id != old.info.trace_id
    assert {span.name for span in traces[0].data.spans} >= {
        "decode_BASELINE",
        "decode_INTERVENTION_ONLY",
    }
    assert len(tracker.client.search_runs([experiment.experiment_id])) == 1
    original = MlflowClient.get_trace
    monkeypatch.setattr(
        MlflowClient,
        "get_trace",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("trace read failed")),
    )
    # Trace failures must retain persisted calculation results for retry.
    with pytest.raises(RuntimeError, match="trace read failed"):
        args.intervention_conditions = ["latent_only"]
        case.execute(wrapper, args)
    monkeypatch.setattr(MlflowClient, "get_trace", original)
    wrapper.model.forward.reset_mock()
    case.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0
