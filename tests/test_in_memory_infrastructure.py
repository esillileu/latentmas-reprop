from pathlib import Path

import pytest
import torch

from latentmas_reprop.domain.models.tensor_artifact import (
    ImmutableTensorOverwriteError,
    TensorMissingError,
)
from latentmas_reprop.infrastructure.storage import InMemoryStorage
from latentmas_reprop.infrastructure.tracking import InMemoryTracker


def test_in_memory_tracker_lifecycle_and_spans():
    tracker = InMemoryTracker()
    run = tracker.start_run("test_exp", run_name="run_01", tags={"env": "test"})
    assert run.info.run_id.startswith("in_memory_")
    assert tracker.active_run_id == run.info.run_id

    tracker.log_params({"lr": 1e-4, "seed": 42})
    tracker.log_metrics({"acc": 0.98}, step=1)
    tracker.log_dict({"config": "custom"}, "cfg.json")
    tracker.log_artifact(Path("README.md"))

    with tracker.start_sample_trace("sample_1", {"input": "hello"}, tags={"task": "gsm8k"}) as root:
        assert root.trace_id is not None
        root.set_inputs({"input": "hello world"})
        with tracker.start_span("child_span", "LLM", {"prompt": "p"}) as child:
            child.set_token_usage(5, 10)
            child.set_outputs({"reply": "ans"})
            child.set_status("OK")
        root.set_outputs({"result": "done"})

    tracker.log_expectation(root.trace_id, "correct", "ans")
    tracker.log_feedback(root.trace_id, "quality", 1.0)
    tracker.end_run("FINISHED")

    assert tracker.active_run_id is None
    record = tracker.runs[run.info.run_id]
    assert record["status"] == "FINISHED"
    assert record["params"]["lr"] == 1e-4
    assert record["metrics"]["acc"] == 0.98
    assert len(tracker.traces) == 1
    assert len(tracker.traces[0]["spans"]) == 2
    assert len(tracker.assessments) == 2


def test_in_memory_storage_save_and_retrieve():
    storage = InMemoryStorage()
    t1 = torch.tensor([1.0, 2.0])
    storage.save_condition("run_a", "s1", "own", {"val": 10}, tensors={"hidden": t1})

    assert storage.is_condition_complete("run_a", "s1", "own") is True
    res = storage.load_condition_result("run_a", "s1", "own")
    assert res["val"] == 10
    assert "hidden.pt" in res["files"]

    loaded_t = storage.load_condition_tensor("run_a", "s1", "own", "hidden")
    assert torch.equal(loaded_t, t1)

    with pytest.raises(ImmutableTensorOverwriteError):
        storage.save_tensor("tensor.pt", t1)
        storage.save_tensor("tensor.pt", torch.tensor([3.0]))

    with pytest.raises(TensorMissingError):
        storage.load_condition_tensor("run_a", "s1", "own", "missing")
