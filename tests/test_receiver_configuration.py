"""Verify model presets produce one run with complete sample trajectories."""

from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from latentmas_reprop.application.receiver_reasoning_use_case import (
    ReceiverReasoningUseCase,
)
from latentmas_reprop.domain.ports.tracking_port import DummySpan, ExperimentTrackerPort
from src.run.cli import parse_args


@pytest.mark.parametrize("preset", ["lm_q34_gsm8k", "lm_q30.6_gsm8k"])
def test_twenty_samples_share_one_run_and_keep_all_conditions(tmp_path, preset):
    args = parse_args(["-c", f"lmas/receiver_reasoning/{preset}"])
    built, roots = [], []

    class Method:
        def __init__(self, model, **kwargs):
            self.step = kwargs["latent_steps"]

        def build_latent_contexts(self, items):
            built.append((items[0]["question"], self.step))
            context = ((torch.full((1, 1, self.step + 20, 1), float(self.step)),),)
            return context, [[{"latent_steps": self.step, "input_ids": [1]}] * 3]

        def decode_with_context(self, items, **kwargs):
            context = kwargs["past_kv"]
            if self.step:
                assert context[0][0].shape[-2] == self.step + 20
                assert torch.all(context[0][0] == self.step)
                context[0][0].zero_()
            else:
                assert context is None
            return [
                {
                    "prediction": "1",
                    "raw_prediction": "1",
                    "correct": True,
                    "agents": [{"input_ids": [1], "generated_tokens": 1}],
                }
            ]

    class SampleSpan(DummySpan):
        def __init__(self, index):
            self.index = index
            self.outputs = {}

        @property
        def trace_id(self):
            return f"sample-trace-{self.index}"

        def set_outputs(self, outputs):
            self.outputs = outputs

    @contextmanager
    def start_sample(name, inputs, **kwargs):
        span = SampleSpan(inputs["sample_index"])
        roots.append(span)
        yield span

    tracker = Mock(spec=ExperimentTrackerPort)
    tracker.start_sample_trace.side_effect = start_sample
    tracker.start_span.side_effect = lambda *a, **kw: ExperimentTrackerPort.start_span(
        tracker, *a, **kw
    )
    dataset = SimpleNamespace(
        load=lambda **kw: [{"question": f"q{i}", "gold": "1"} for i in range(25)]
    )
    with patch(
        "latentmas_reprop.application.receiver_reasoning.runner.LatentMASMethod", Method
    ):
        summary, records = ReceiverReasoningUseCase(
            dataset_port=dataset,
            cache_port=SimpleNamespace(get_layer_path=lambda layer: tmp_path),
            tracker_port=tracker,
        ).execute(SimpleNamespace(device="cpu"), args)

    tracker.start_run.assert_called_once()
    tracker.end_run.assert_called_once_with(status="FINISHED")
    assert len(roots) == 20
    assert len(built) == 40
    assert len(records) == 120
    assert summary["execution"]["n_records_expected"] == 120
    expected = {
        (step, mode)
        for step in [0, *args.upstream_steps]
        for mode in ("answer_only", "free")
    }
    trajectories = defaultdict(list)
    for row in records:
        trajectories[row["trace_id"]].append(row)
    assert len(trajectories) == 20
    for root in roots:
        rows = trajectories[root.trace_id]
        assert {
            (r["upstream_latent_steps"], r["receiver_mode"]) for r in rows
        } == expected
        assert len({r["sample_id"] for r in rows}) == 1
        assert root.outputs["execution"]["n_records_completed"] == 6
        assert len(root.outputs["cells"]) == 6
        for step in [0, *args.upstream_steps]:
            assert (
                len(
                    {
                        r["context_id"]
                        for r in rows
                        if r["upstream_latent_steps"] == step
                    }
                )
                == 1
            )
