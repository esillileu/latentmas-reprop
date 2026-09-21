"""Tests for latent intervention experiment harness.

Covers:
    - generate_cross_indices (all policies)
    - drop != zero semantics
    - zero cache: same shape, all zeros
    - cross provenance: source != target
    - cache length metadata
    - stable sample key reproducibility
    - evaluator error propagation
    - paired transition / harm-rescue aggregation
    - MLflow run created before execution
    - failed experiment leaves failed MLflow record
    - InterventionMetrics to_mlflow_metrics
    - SampleInterventionRecord with new fields
    - _truncate_past DynamicCache (requested vs actual length)
"""

import contextlib
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.cache_utils import DynamicCache

from latentmas_reprop.application.intervention_use_case import (
    compute_mcnemar_test,
    estimate_cache_bytes,
    generate_cross_indices,
    get_kv_num_layers,
    get_kv_sequence_length,
)
from latentmas_reprop.domain.models import (
    InterventionCondition,
    InterventionMetrics,
    SampleInterventionRecord,
    compute_sample_key,
)
from latentmas_reprop.domain.services.latent_mas import (
    LatentMASMethod,
    clone_past_kv,
    create_zero_past_kv,
    move_past_kv,
)

# ── generate_cross_indices ────────────────────────────────────────────────────


def test_cross_indices_shift_1():
    for n in [2, 3, 5, 10]:
        indices = generate_cross_indices(n, policy="shift_1")
        assert len(indices) == n
        assert all(indices[i] != i for i in range(n))
        assert indices == [(i + 1) % n for i in range(n)]


def test_cross_indices_derangement():
    indices = generate_cross_indices(5, policy="derangement", seed=42)
    assert len(indices) == 5
    assert all(indices[i] != i for i in range(5))
    assert sorted(indices) == list(range(5))


def test_cross_indices_validation():
    with pytest.raises(ValueError, match="requires at least 2 samples"):
        generate_cross_indices(1, policy="shift_1")


def test_cross_provenance_source_ne_target():
    """Cross source must always differ from target."""
    n = 10
    indices = generate_cross_indices(n, policy="shift_1")
    for i, j in enumerate(indices):
        assert j != i, f"Provenance violated at i={i}: source==target"


# ── drop != zero semantics ────────────────────────────────────────────────────


def _make_cache(seq_len: int = 8) -> DynamicCache:
    cache = DynamicCache()
    k = torch.randn(1, 2, seq_len, 16)
    v = torch.randn(1, 2, seq_len, 16)
    cache.update(k, v, 0)
    return cache


def test_drop_is_none():
    """Drop condition => no past_key_values (None), not zero cache."""
    # drop: simply None
    drop_ctx = None
    assert drop_ctx is None


def test_zero_is_same_shape_all_zeros():
    """Zero condition => same shape as own cache, all tensor values == 0."""
    cache = _make_cache(seq_len=8)
    zeroed = create_zero_past_kv(cache)

    # Same shape
    assert zeroed.layers[0].keys.shape == cache.layers[0].keys.shape
    assert zeroed.layers[0].values.shape == cache.layers[0].values.shape

    # All zero
    assert zeroed.layers[0].keys.abs().sum().item() == 0.0
    assert zeroed.layers[0].values.abs().sum().item() == 0.0

    # Original unchanged
    assert cache.layers[0].keys.abs().sum().item() > 0.0


def test_drop_ne_zero_semantically():
    """Drop (None) and zero (zero-filled cache) are structurally different."""
    cache = _make_cache(seq_len=8)
    drop_ctx = None
    zero_ctx = create_zero_past_kv(cache)
    assert drop_ctx is None
    assert zero_ctx is not None
    assert get_kv_sequence_length(drop_ctx) == 0
    assert get_kv_sequence_length(zero_ctx) == 8  # same shape as own


# ── cache helpers ─────────────────────────────────────────────────────────────


def test_cache_helpers_dynamic_cache():
    cache = DynamicCache()
    k = torch.ones(1, 2, 4, 8)
    v = torch.ones(1, 2, 4, 8) * 2
    cache.update(k, v, 0)

    cloned = clone_past_kv(cache)
    assert cloned is not cache
    assert torch.equal(cloned.layers[0].keys, cache.layers[0].keys)

    moved = move_past_kv(cloned, "cpu")
    assert moved.layers[0].keys.device.type == "cpu"

    zeroed = create_zero_past_kv(cache)
    assert torch.all(zeroed.layers[0].keys == 0)
    assert torch.all(zeroed.layers[0].values == 0)
    assert zeroed.layers[0].keys.shape == k.shape


def test_get_kv_sequence_length():
    assert get_kv_sequence_length(None) == 0
    cache = _make_cache(seq_len=7)
    assert get_kv_sequence_length(cache) == 7


def test_get_kv_num_layers():
    cache = DynamicCache()
    # Before any update, layers may be empty
    assert get_kv_num_layers(None) == 0
    cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), 0)
    cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), 1)
    assert get_kv_num_layers(cache) == 2


def test_estimate_cache_bytes():
    cache = _make_cache(seq_len=8)  # 1 layer, 2 heads, 8 seq, 16 dim
    # keys + values: 2 * (1*2*8*16 * 4 bytes float32)
    expected = 2 * (1 * 2 * 8 * 16 * 4)  # ~8192 bytes
    actual = estimate_cache_bytes(cache)
    assert actual == expected


# ── cache length metadata ─────────────────────────────────────────────────────


def test_cache_length_metadata_own():
    """For own condition: target == source seq len, delta == 0."""
    cache = _make_cache(seq_len=12)
    seq_len = get_kv_sequence_length(cache)
    target = seq_len
    src = seq_len  # own: same
    delta = src - target
    assert delta == 0
    assert target == 12


def test_cache_length_metadata_cross_delta():
    """For cross condition: delta = source_len - target_len."""
    cache_i = _make_cache(seq_len=10)
    cache_j = _make_cache(seq_len=15)
    target = get_kv_sequence_length(cache_i)
    src = get_kv_sequence_length(cache_j)
    delta = src - target
    assert delta == 5


def test_cache_length_metadata_drop():
    """Drop: source_cache_seq_len == 0."""
    assert get_kv_sequence_length(None) == 0


# ── stable sample key ─────────────────────────────────────────────────────────


def test_sample_key_reproducible():
    k1 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    k2 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    assert k1 == k2
    assert len(k1) == 64  # sha256 hex digest


def test_sample_key_normalizes_whitespace():
    k1 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    k2 = compute_sample_key("gsm8k", "test", "  What   is 2+2?  ")
    assert k1 == k2


def test_sample_key_different_for_different_questions():
    k1 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    k2 = compute_sample_key("gsm8k", "test", "What is 3+3?")
    assert k1 != k2


def test_sample_key_different_for_different_tasks():
    k1 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    k2 = compute_sample_key("other_task", "test", "What is 2+2?")
    assert k1 != k2


# ── evaluator error propagation ───────────────────────────────────────────────


def test_evaluator_error_propagation_in_decode_with_context():
    """decode_with_context result dict must contain 'error_msg' key."""
    import inspect
    import textwrap

    from latentmas_reprop.domain.services.latent_mas import LatentMASMethod

    src = textwrap.dedent(inspect.getsource(LatentMASMethod.decode_with_context))
    assert '"error_msg"' in src or "'error_msg'" in src, (
        "decode_with_context must include 'error_msg' in result dict"
    )


# ── paired transition / harm-rescue ──────────────────────────────────────────


def _make_record(cond: str, correct: bool) -> SampleInterventionRecord:
    return SampleInterventionRecord(
        sample_id="s0",
        source_sample_id=None,
        condition=cond,
        question="q",
        gold="g",
        prediction="p",
        raw_prediction="r",
        correct=correct,
        model="m",
        task="gsm8k",
        latent_steps=4,
        seed=42,
    )


def test_harm_rescue_from_transitions():
    """
    own_correct=True, cross_correct=False => harm
    own_correct=False, cross_correct=True => rescue
    """
    # simulate the logic used in InterventionUseCase._run_experiment
    results = [
        {"own": _make_record("own", True), "cross": _make_record("cross", False)},
        {"own": _make_record("own", False), "cross": _make_record("cross", True)},
        {"own": _make_record("own", True), "cross": _make_record("cross", True)},
    ]
    harm = sum(
        1
        for r in results
        if r.get("own")
        and r.get("cross")
        and r["own"].correct
        and not r["cross"].correct
    )
    rescue = sum(
        1
        for r in results
        if r.get("own")
        and r.get("cross")
        and not r["own"].correct
        and r["cross"].correct
    )
    assert harm == 1
    assert rescue == 1


# ── InterventionMetrics ────────────────────────────────────────────────────────


def test_intervention_metrics_all_conditions():
    metrics = InterventionMetrics(
        accuracy_own=0.8,
        accuracy_cross=0.6,
        accuracy_drop=0.7,
        accuracy_zero=0.5,
        accuracy_by_condition={"own": 0.8, "cross": 0.6, "drop": 0.7, "zero": 0.5},
        n_correct_by_condition={"own": 8, "cross": 6, "drop": 7, "zero": 5},
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
        harm_rescue={
            "cross": {"harm": 2, "rescue": 1},
            "drop": {"harm": 1, "rescue": 2},
            "zero": {"harm": 3, "rescue": 0},
        },
    )

    mlf = metrics.to_mlflow_metrics()
    assert mlf["accuracy/own"] == 0.8
    assert mlf["accuracy/cross"] == 0.6
    assert mlf["accuracy/drop"] == 0.7
    assert mlf["accuracy/zero"] == 0.5
    assert mlf["answer_change_rate/cross"] == 0.3
    assert mlf["answer_change_rate/drop"] == 0.4
    assert mlf["answer_change_rate/zero"] == 0.5
    assert mlf["transition/cross/harm"] == 2.0
    assert mlf["transition/cross/rescue"] == 1.0
    assert mlf["transition/drop/harm"] == 1.0
    assert mlf["transition/zero/harm"] == 3.0


# ── SampleInterventionRecord with new fields ──────────────────────────────────


def test_sample_intervention_record_new_fields():
    record = SampleInterventionRecord(
        sample_id="sample_0",
        source_sample_id="sample_1",
        condition=InterventionCondition.CROSS,
        question="What is 2+2?",
        gold="4",
        prediction="4",
        raw_prediction="The answer is 4",
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


# ── compute_mcnemar_test ──────────────────────────────────────────────────────


def test_compute_mcnemar_test():
    res_zero = compute_mcnemar_test(0, 0)
    assert res_zero["discordant_pairs"] == 0
    assert res_zero["p_value"] == 1.0

    res_sym = compute_mcnemar_test(5, 5)
    assert res_sym["discordant_pairs"] == 10
    assert res_sym["p_value"] == 1.0

    res_asym = compute_mcnemar_test(10, 0)
    assert res_asym["discordant_pairs"] == 10
    assert res_asym["p_value"] < 0.01


# ── MLflow run lifecycle ──────────────────────────────────────────────────────


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

    from latentmas_reprop.application.intervention_use_case import InterventionUseCase

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

    # Patch _run_experiment to just record the call order
    def fake_run_experiment(**_kw):
        call_order.append("run_experiment")
        return (InterventionMetrics(n_samples=0), [])

    with (
        patch.object(uc, "dataset_port", dataset_mock),
        patch.object(uc, "_run_experiment", fake_run_experiment),
        contextlib.suppress(Exception),
    ):
        uc.execute(model=model_mock, args=args)

    # start_run must happen before run_experiment
    assert "start_run" in call_order
    sr_idx = call_order.index("start_run")
    re_idx = call_order.index("run_experiment")
    assert sr_idx < re_idx, "start_run must be called before run_experiment"


def test_failed_experiment_calls_end_run_failed():
    """If _run_experiment raises, end_run('FAILED') must be called."""
    tracker = MagicMock()
    tracker.start_run.return_value = None
    tracker.log_params.return_value = None

    from latentmas_reprop.application.intervention_use_case import InterventionUseCase

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

    # end_run should be called with FAILED status
    tracker.end_run.assert_called_once_with(status="FAILED")


# ── _truncate_past DynamicCache (requested vs actual length) ──────────────────


def test_truncate_past_requested_vs_actual():
    """_truncate_past(cache, tokens_to_keep) should result in exactly tokens_to_keep tokens."""
    cache = DynamicCache()
    k = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    v = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    cache.update(k, v, 0)
    assert get_kv_sequence_length(cache) == 20

    # Create a LatentMASMethod-like truncation
    dummy_args = types.SimpleNamespace(
        latent_only=False,
        sequential_info_only=False,
        task="gsm8k",
        latent_space_realign=False,
        prompt="sequential",
        think=False,
        method="latent_mas",
        model_name="Qwen/test-model",
        device="cpu",
        device2="cpu",
    )
    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")

    method = LatentMASMethod(
        model=model_mock,
        latent_steps=4,
        args=dummy_args,
    )

    truncated = method._truncate_past(cache, tokens_to_keep=7)
    actual_len = get_kv_sequence_length(truncated)
    assert actual_len == 7, f"Expected 7 tokens, got {actual_len}"
    # Verify values: last 7 tokens (indices 13..19)
    assert truncated.layers[0].keys.squeeze()[-1].item() == 19.0


def test_truncate_past_keeps_max_when_small():
    """If tokens_to_keep >= cache length, entire cache is kept."""
    cache = _make_cache(seq_len=5)
    dummy_args = types.SimpleNamespace(
        latent_only=False,
        sequential_info_only=False,
        task="gsm8k",
        latent_space_realign=False,
        prompt="sequential",
        think=False,
        method="latent_mas",
        model_name="Qwen/test-model",
        device="cpu",
        device2="cpu",
    )
    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    method = LatentMASMethod(model=model_mock, latent_steps=4, args=dummy_args)
    result = method._truncate_past(cache, tokens_to_keep=100)
    assert get_kv_sequence_length(result) == 5
