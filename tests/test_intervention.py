import pytest
import torch
from transformers.cache_utils import DynamicCache

from latentmas_reprop.application.intervention_use_case import generate_cross_indices
from latentmas_reprop.domain.models import (
    InterventionCondition,
    InterventionMetrics,
    SampleInterventionRecord,
)
from latentmas_reprop.domain.services.latent_mas import (
    clone_past_kv,
    create_zero_past_kv,
    move_past_kv,
)


def test_cross_indices_shift_1():
    for n in [2, 3, 5, 10]:
        indices = generate_cross_indices(n, policy="shift_1")
        assert len(indices) == n
        # Check derangement: no fixed point
        assert all(indices[i] != i for i in range(n))
        # Check shift
        assert indices == [(i + 1) % n for i in range(n)]


def test_cross_indices_derangement():
    indices = generate_cross_indices(5, policy="derangement", seed=42)
    assert len(indices) == 5
    assert all(indices[i] != i for i in range(5))
    # Must be a valid permutation
    assert sorted(indices) == list(range(5))


def test_cross_indices_validation():
    with pytest.raises(ValueError, match="requires at least 2 samples"):
        generate_cross_indices(1, policy="shift_1")


def test_cache_helpers_dynamic_cache():
    cache = DynamicCache()
    k = torch.ones(1, 2, 4, 8)
    v = torch.ones(1, 2, 4, 8) * 2
    cache.update(k, v, 0)

    # Test clone
    cloned = clone_past_kv(cache)
    assert cloned is not cache
    assert torch.equal(cloned.layers[0].keys, cache.layers[0].keys)

    # Test move
    moved = move_past_kv(cloned, "cpu")
    assert moved.layers[0].keys.device.type == "cpu"

    # Test zero
    zeroed = create_zero_past_kv(cache)
    assert torch.all(zeroed.layers[0].keys == 0)
    assert torch.all(zeroed.layers[0].values == 0)
    assert zeroed.layers[0].keys.shape == k.shape


def test_intervention_metrics():
    metrics = InterventionMetrics(
        accuracy_own=1.0,
        accuracy_cross=0.5,
        accuracy_zero=0.25,
        answer_change_rate_cross=0.5,
        answer_change_rate_zero=0.75,
        accuracy_delta_own_cross=0.5,
        accuracy_delta_own_zero=0.75,
        n_samples=4,
        n_own_correct=4,
        n_cross_correct=2,
        n_zero_correct=1,
        runtime_total=10.0,
        runtime_per_sample=2.5,
    )

    mlflow_dict = metrics.to_mlflow_metrics()
    assert mlflow_dict["accuracy/own"] == 1.0
    assert mlflow_dict["accuracy/cross"] == 0.5
    assert mlflow_dict["accuracy/zero"] == 0.25
    assert mlflow_dict["answer_change_rate/cross"] == 0.5
    assert mlflow_dict["accuracy_delta/own_cross"] == 0.5
    assert mlflow_dict["n_samples"] == 4.0


def test_sample_intervention_record():
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
        latency=0.12,
        generated_tokens=10,
    )

    d = record.to_dict()
    assert d["sample_id"] == "sample_0"
    assert d["source_sample_id"] == "sample_1"
    assert d["condition"] == "cross"
    assert d["correct"] is True
