"""Tests for latent intervention use case and tracking metrics."""

import contextlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.domain.models import (
    InterventionMetrics,
    SampleInterventionRecord,
)


def test_evaluator_error_propagation_in_decode_with_context():
    """decode_with_context result dict must contain 'error_msg' key."""
    import inspect
    import textwrap

    from latentmas_reprop.domain.services.latent_mas import LatentMASMethod

    src = textwrap.dedent(inspect.getsource(LatentMASMethod.decode_with_context))
    assert '"error_msg"' in src or "'error_msg'" in src, (
        "decode_with_context must include 'error_msg' in result dict"
    )


def test_intervention_metrics_all_conditions():
    metrics = InterventionMetrics(
        accuracy_own=0.8,
        accuracy_cross=0.6,
        accuracy_drop=0.7,
        accuracy_zero=0.5,
        answer_change_rate_cross=0.3,
        answer_change_rate_drop=0.4,
        answer_change_rate_zero=0.5,
        answer_change_rate_by_condition={"cross": 0.3, "drop": 0.4, "zero": 0.5},
        accuracy_delta_own_cross=0.2,
        accuracy_delta_own_drop=0.1,
        accuracy_delta_own_zero=0.3,
        n_samples=10,
        n_own_correct=8,
        n_cross_correct=6,
        n_drop_correct=7,
        n_zero_correct=5,
        runtime_total=10.0,
        runtime_per_sample=1.0,
        runtime_own=2.5,
        runtime_cross=2.5,
        runtime_drop=2.5,
        runtime_zero=2.5,
    )

    mlf = metrics.to_mlflow_metrics()
    assert mlf["accuracy/own"] == 0.8
    assert mlf["accuracy/cross"] == 0.6
    assert mlf["accuracy/drop"] == 0.7
    assert mlf["accuracy/zero"] == 0.5
    assert mlf["answer_change_rate/cross"] == 0.3
    assert mlf["answer_change_rate/drop"] == 0.4
    assert mlf["answer_change_rate/zero"] == 0.5


def test_sample_intervention_record_schema():
    record = SampleInterventionRecord(
        sample_id="sample_0",
        source_sample_id="sample_1",
        condition="cross",
        question="What is 2+2?",
        gold="4",
        prediction="4",
        raw_prediction="The answer is 4.",
        correct=True,
        model="test-model",
        task="gsm8k",
        latent_steps=4,
        seed=42,
        sample_index=0,
        sample_key="abc123",
        source_sample_index=1,
        source_sample_key="def456",
        target_cache_seq_len=20,
        source_cache_seq_len=22,
        cache_seq_len_delta=2,
        cache_present=True,
        num_layers=16,
        cache_dtype="torch.bfloat16",
        latency=0.12,
        generated_tokens=10,
    )

    d = record.to_dict()
    assert d["sample_key"] == "abc123"
    assert d["source_sample_key"] == "def456"
    assert d["target_cache_seq_len"] == 20
    assert d["source_cache_seq_len"] == 22
    assert d["cache_seq_len_delta"] == 2
    assert d["num_layers"] == 16
    assert d["condition"] == "cross"
    assert d["correct"] is True


def test_mlflow_run_created_before_execution():
    """Tracker.start_run must be called before decode_with_context is called."""
    call_order: list[str] = []

    tracker = MagicMock()
    tracker.start_run.side_effect = lambda **_kw: call_order.append("start_run")
    tracker.log_params.return_value = None
    tracker.start_sample_trace.return_value.__enter__ = lambda s: (
        call_order.append("trace_enter") or MagicMock()
    )
    tracker.start_sample_trace.return_value.__exit__ = lambda s, *a: False

    uc = InterventionUseCase(tracker_port=tracker)

    dataset_mock = MagicMock()
    dataset_mock.load.return_value = [
        {"question": "q0", "gold": "1"},
        {"question": "q1", "gold": "2"},
    ]

    args = types.SimpleNamespace(
        method="latent_mas",
        task="gsm8k",
        split="test",
        max_samples=2,
        latent_steps=4,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        seed=42,
        model_name="test-model",
        tracking_experiment_name="test_exp",
        prompt="sequential",
        use_vllm=False,
        save_raw_cache=False,
        cross_policy="shift_1",
        intervention_conditions="own,cross,drop,zero",
    )

    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    model_mock.model = MagicMock()
    model_mock.model.dtype = "bfloat16"

    def fake_run_experiment(**_kw):
        call_order.append("run_experiment")
        return (InterventionMetrics(n_samples=0), [])

    with (
        patch.object(uc, "dataset_port", dataset_mock),
        patch.object(uc, "_run_experiment", fake_run_experiment),
        contextlib.suppress(Exception),
    ):
        uc.execute(model=model_mock, args=args)

    assert "start_run" in call_order
    sr_idx = call_order.index("start_run")
    re_idx = call_order.index("run_experiment")
    assert sr_idx < re_idx, "start_run must be called before run_experiment"


def test_failed_experiment_calls_end_run_failed():
    """If _run_experiment raises, end_run('FAILED') must be called."""
    tracker = MagicMock()
    tracker.start_run.return_value = None
    tracker.log_params.return_value = None

    uc = InterventionUseCase(tracker_port=tracker)

    dataset_mock = MagicMock()
    dataset_mock.load.return_value = [
        {"question": "q0", "gold": "1"},
        {"question": "q1", "gold": "2"},
    ]

    args = types.SimpleNamespace(
        method="latent_mas",
        task="gsm8k",
        split="test",
        max_samples=2,
        latent_steps=4,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        seed=42,
        model_name="test-model",
        tracking_experiment_name="test_exp",
        prompt="sequential",
        use_vllm=False,
        save_raw_cache=False,
        cross_policy="shift_1",
        intervention_conditions="own,cross,drop,zero",
    )

    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    model_mock.model = MagicMock()
    model_mock.model.dtype = "bfloat16"

    def boom(**_kw):
        raise RuntimeError("Simulated OOM")

    with (
        patch.object(uc, "dataset_port", dataset_mock),
        patch.object(uc, "_run_experiment", boom),
        pytest.raises(RuntimeError, match="Simulated OOM"),
    ):
        uc.execute(model=model_mock, args=args)

    tracker.end_run.assert_called_once_with(status="FAILED")
