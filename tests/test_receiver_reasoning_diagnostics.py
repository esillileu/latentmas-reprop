import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from latentmas_reprop.application.receiver_reasoning.diagnostics import cell_diagnostics
from latentmas_reprop.application.receiver_reasoning.runtime import RuntimeMeasurements
from latentmas_reprop.application.receiver_reasoning_use_case import (
    ReceiverReasoningUseCase,
)
from latentmas_reprop.domain.ports.tracking_port import DummySpan, ExperimentTrackerPort
from src.run.cli import parse_args


def test_gpu_measurement_synchronizes_and_reads_model_device():
    model = SimpleNamespace(device="cuda:1")
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize") as sync,
        patch("torch.cuda.max_memory_allocated", return_value=123) as allocated,
        patch("torch.cuda.max_memory_reserved", return_value=456) as reserved,
        patch(
            "latentmas_reprop.application.receiver_reasoning.runtime.perf_counter",
            side_effect=[10.0, 12.5],
        ),
    ):
        with (
            patch("torch.cuda.memory_allocated", return_value=100),
            patch("torch.cuda.memory_reserved", return_value=200),
            patch("torch.cuda.reset_peak_memory_stats") as reset,
        ):
            runtime = RuntimeMeasurements(model)
            span = Mock(spec=DummySpan)
            with runtime.measure(span) as measurement:
                pass
            assert measurement["latency_sec"] == 2.5
            assert measurement["peak_vram_bytes"] == 123
            assert measurement["incremental_peak_vram_bytes"] == 23
            assert str(reset.call_args.args[0]) == "cuda:1"
            span.set_attribute.assert_any_call("peak_vram_bytes", 123)
        assert sync.call_count == 2
        assert str(sync.call_args.args[0]) == "cuda:1"
        assert runtime.run_peak() == {
            "peak_vram_bytes": 123,
            "peak_vram_reserved_bytes": 456,
        }
        assert str(allocated.call_args.args[0]) == "cuda:1"
        assert str(reserved.call_args.args[0]) == "cuda:1"


def test_unrecorded_legacy_measurements_remain_absent():
    assert cell_diagnostics([{"receiver_generated_tokens": 4}]) == {}


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failure_preserves_partial_records_and_experimental_diagnostics(
    tmp_path, error_type
):
    args = parse_args(["-c", "lmas/receiver_reasoning/gsm8k", "--max_samples", "1"])

    class Method:
        def __init__(self, *args, **kwargs):
            pass

        def decode_with_context(self, items, **kwargs):
            if kwargs["receiver_mode"] == "free":
                raise error_type("receiver failed")
            return [
                {
                    "prediction": "",
                    "raw_prediction": "",
                    "correct": False,
                    "error_msg": "parse failed",
                    "agents": [{"input_ids": [1], "generated_tokens": 3}],
                }
            ]

    tracker = Mock(spec=ExperimentTrackerPort)
    # Exercise the port's no-op tracing implementation, independent of remote I/O.
    tracker.start_sample_trace.side_effect = lambda *a, **kw: (
        ExperimentTrackerPort.start_sample_trace(tracker, *a, **kw)
    )
    tracker.start_span.side_effect = lambda *a, **kw: ExperimentTrackerPort.start_span(
        tracker, *a, **kw
    )
    dataset = SimpleNamespace(load=lambda **kw: [{"question": "q", "gold": "1"}])
    use_case = ReceiverReasoningUseCase(
        dataset_port=dataset,
        cache_port=SimpleNamespace(get_layer_path=lambda layer: tmp_path),
        tracker_port=tracker,
    )
    with (
        patch(
            "latentmas_reprop.application.receiver_reasoning.runner.LatentMASMethod",
            Method,
        ),
        pytest.raises(error_type, match="receiver failed"),
    ):
        use_case.execute(SimpleNamespace(device="cpu", load_time_sec=1.25), args)
    failure = json.loads(next(tmp_path.glob("*/failure.json")).read_text())
    assert "receiver failed" in failure["failure_traceback"]
    assert failure["execution"]["n_records_expected"] == 6
    assert failure["execution"]["n_records_completed"] == 1
    assert failure["execution"]["n_errors"] == 1
    assert failure["execution"]["parse_failure_count"] == 1
    assert failure["execution"]["empty_output_count"] == 1
    assert failure["runtime"]["model_load_time_sec"] == 1.25
    assert (
        len(next(tmp_path.glob("*/sample_results.jsonl")).read_text().splitlines()) == 1
    )
    tracker.end_run.assert_called_once_with(
        status="KILLED" if error_type is KeyboardInterrupt else "FAILED"
    )
    names = {call.args[0].name for call in tracker.log_artifact.call_args_list}
    assert names == {"failure.json", "resolved_config.yaml", "sample_results.jsonl"}
    assert tracker.log_metrics.call_args.args[0]["execution_n_records_completed"] == 1


@pytest.fixture
def cuda_counters():
    state = {
        "allocated": 100,
        "reserved": 200,
        "peak_allocated": 700,
        "peak_reserved": 900,
    }

    def reset(device):
        state["peak_allocated"] = state["allocated"]
        state["peak_reserved"] = state["reserved"]

    def allocate(allocated, reserved):
        state["allocated"], state["reserved"] = allocated, reserved
        state["peak_allocated"] = max(state["peak_allocated"], allocated)
        state["peak_reserved"] = max(state["peak_reserved"], reserved)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.synchronize"),
        patch(
            "torch.cuda.max_memory_allocated",
            side_effect=lambda device: state["peak_allocated"],
        ),
        patch(
            "torch.cuda.max_memory_reserved",
            side_effect=lambda device: state["peak_reserved"],
        ),
        patch(
            "torch.cuda.memory_allocated", side_effect=lambda device: state["allocated"]
        ),
        patch(
            "torch.cuda.memory_reserved", side_effect=lambda device: state["reserved"]
        ),
        patch("torch.cuda.reset_peak_memory_stats", side_effect=reset),
    ):
        yield RuntimeMeasurements(SimpleNamespace(device="cuda:1")), allocate


def test_nested_span_peaks_do_not_inherit_previous_scopes(cuda_counters):
    runtime, allocate = cuda_counters
    with runtime.measure(DummySpan()) as first_sample:
        allocate(110, 210)
        with runtime.measure(DummySpan()) as upstream:
            allocate(500, 600)
            allocate(120, 220)
        with runtime.measure(DummySpan()) as receiver:
            allocate(250, 350)
            allocate(100, 200)
    assert first_sample["peak_vram_bytes"] == upstream["peak_vram_bytes"] == 500
    assert receiver["peak_vram_bytes"] == 250
    assert receiver["incremental_peak_vram_bytes"] == 130
    with runtime.measure(DummySpan()) as second_sample:
        allocate(180, 280)
    assert second_sample["peak_vram_bytes"] == 180
    assert second_sample["incremental_peak_vram_bytes"] == 80
    assert runtime.run_peak() == {
        "peak_vram_bytes": 700,
        "peak_vram_reserved_bytes": 900,
    }


def test_failed_span_retains_peak_and_parent_accumulates_it(cuda_counters):
    runtime, allocate = cuda_counters
    child = Mock(spec=DummySpan)
    with (
        pytest.raises(RuntimeError, match="inference failed"),
        runtime.measure(DummySpan()) as sample,
        runtime.measure(child),
    ):
        allocate(800, 950)
        allocate(100, 200)
        raise RuntimeError("inference failed")
    child.set_attribute.assert_any_call("peak_vram_bytes", 800)
    assert sample["peak_vram_bytes"] == 800
    assert runtime.run_peak() == {
        "peak_vram_bytes": 800,
        "peak_vram_reserved_bytes": 950,
    }
