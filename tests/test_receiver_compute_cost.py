"""Cost measurements account for handoff length without extrapolating latency."""

from itertools import permutations

import pytest
import torch

from latentmas_reprop.application.receiver_compute_preflight.cost import (
    ReceiverTiming,
    compute_proxy,
    prefix_cost,
)
from latentmas_reprop.application.receiver_compute_preflight.inference import (
    length_matched_donors,
)


def test_minimum_length_derangement_matches_brute_force_optimum():
    for lengths in ([10, 5, 10, 40, 6], [1, 2, 100, 101], [7, 7, 8, 8], [1, 2, 3]):
        donors = length_matched_donors(lengths)
        observed = sum(abs(lengths[i] - lengths[d]) for i, d in enumerate(donors))
        optimum = min(
            sum(abs(lengths[i] - lengths[d]) for i, d in enumerate(p))
            for p in permutations(range(len(lengths)))
            if all(i != d for i, d in enumerate(p))
        )
        assert observed == optimum
        assert len(set(donors)) == len(lengths)
    assert length_matched_donors([1, 2, 100, 101]) == [1, 0, 3, 2]


def test_proxy_counts_prefill_and_decode_attention():
    cost = compute_proxy(10, 3, 4)
    assert (
        cost["receiver_attention_pairs"]
        == (10 + 1) + (10 + 2) + (10 + 3) + 14 + 15 + 16
    )
    assert cost["receiver_processed_positions"] == 6
    assert (
        compute_proxy(100, 3, 4)["receiver_attention_pairs"]
        > cost["receiver_attention_pairs"]
    )
    assert compute_proxy(10, 3, 1)["receiver_attention_pairs"] == 36


def test_synchronized_checkpoints_observe_without_stopping(monkeypatch):
    import latentmas_reprop.application.receiver_compute_preflight.cost as module

    times = iter([10.0, 12.0, 15.0, 16.0])
    monkeypatch.setattr(module, "perf_counter", lambda: next(times))
    timing = ReceiverTiming(3, [2, 4], "cuda")
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    timing.start()
    assert timing(torch.ones(1, 4), None) is False
    assert timing(torch.ones(1, 5), None) is False
    assert timing(torch.ones(1, 7), None) is False
    assert timing.checkpoints == {2: 2.0, 4: 5.0}
    assert timing.elapsed() == 6.0
    assert len(calls) == 4


def test_prefix_cost_uses_measured_checkpoint_or_natural_end():
    metadata = {
        "receiver_budget_latency_sec": {"64": 2, "128": 3},
        "receiver_latency_sec": 10,
        "receiver_cache_positions": 100,
        "receiver_prompt_tokens": 20,
    }
    assert prefix_cost(metadata, 64, 64)["receiver_latency_sec"] == 2
    assert prefix_cost(metadata, 90, 128)["receiver_latency_sec"] == 10
    assert prefix_cost(metadata, 90, "free")["receiver_latency_sec"] == 10
    with pytest.raises(KeyError):
        prefix_cost(metadata, 256, 256)


def test_primary_cost_curves_and_target_costs_have_paired_ci():
    from test_receiver_compute_preflight import config_and_records

    from latentmas_reprop.application.receiver_compute_preflight.analysis import analyze

    config, rows = config_and_records()
    metrics, statistics = analyze(rows, config)
    first = metrics["curves"][0]
    assert first["mean_receiver_latency_sec"] == pytest.approx(0.1)
    assert first["receiver_latency_sec_ci_low"] == pytest.approx(0.1)
    difference = next(
        r for r in metrics["cost_comparisons"] if r["metric"] == "receiver_latency_sec"
    )
    assert difference["difference"] == pytest.approx(-0.1)
    assert statistics["distributions"][
        "10:pairing_gain:2:receiver_latency_sec"
    ] == pytest.approx([-0.1] * config["bootstrap_count"])
    threshold = metrics["thresholds"][0]
    assert threshold["minimum_receiver_budget"] == 2
    assert threshold["mean_receiver_latency_sec_at_minimum_budget"] == pytest.approx(
        0.1
    )
    assert threshold["minimum_mean_receiver_attention_pairs"] == 100
