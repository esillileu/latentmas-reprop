"""Tests for intervention pairing algorithms, sample key hashing, and McNemar test."""

import pytest

from latentmas_reprop.application.intervention.algo import generate_cross_indices
from latentmas_reprop.application.intervention.stats import compute_mcnemar_test
from latentmas_reprop.domain.models import (
    SampleInterventionRecord,
    compute_sample_key,
)


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


def test_sample_key_reproducible():
    k1 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    k2 = compute_sample_key("gsm8k", "test", "What is 2+2?")
    assert k1 == k2
    assert len(k1) == 64


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
