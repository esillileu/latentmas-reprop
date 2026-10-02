"""Prevent future-state leakage, relaxed parity, and unpaired receiver summaries."""

import copy
import csv
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from latentmas_reprop.application.receiver_acquisition import parity
from latentmas_reprop.application.receiver_acquisition.sampling import (
    generate_secret_digit_samples,
)
from latentmas_reprop.application.receiver_acquisition.trajectory import (
    latent_prefix,
    summarize_trajectory,
)


def bundle(steps, dynamic=False):
    keys = torch.arange(3 + steps).reshape(1, 1, 3 + steps, 1).float()
    values = keys + 100
    full = ((keys, values),)
    latent = ((keys[..., 3:, :], values[..., 3:, :]),)
    if dynamic:
        full, latent = DynamicCache(), DynamicCache()
        full.update(keys, values, 0)
        latent.update(keys[..., 3:, :], values[..., 3:, :], 0)
    return SimpleNamespace(
        full=full, prompt_len=3, full_len=3 + steps, latent_only=latent
    )


@pytest.mark.parametrize("dynamic", [False, True])
def test_prefix_excludes_prompt_and_future_positions_without_mutating_full20(dynamic):
    full = bundle(20, dynamic)
    original = copy.deepcopy(full.full)
    for step in range(1, 21):
        prefix = latent_prefix(full, step)
        assert parity.caches_equal(prefix, bundle(step, dynamic).latent_only)
    assert parity.caches_equal(full.full, original)
    with pytest.raises(ValueError):
        latent_prefix(full, 21)


@pytest.mark.parametrize("mismatch", [None, "cache", "logits", "history", "prediction"])
def test_exact_parity_refuses_any_cache_output_or_historical_difference(
    monkeypatch, mismatch
):
    samples = generate_secret_digit_samples(100, 42)
    representatives = [next(s for s in samples if s.digit == d) for d in range(10)]

    def build(model, args, sample):
        result = bundle(args.latent_steps)
        if mismatch == "cache" and args.latent_steps != 20:
            result.latent_only[0][0].add_(1)
        return result

    def observe(model, sample, step, cache, full_len, receiver):
        return {
            "sample_key": sample.sample_key,
            "target_digit": sample.digit,
            "predicted_digit": 5,
            "candidate_log_probabilities": {"5": -0.1},
            "receiver_position_start": full_len,
            "original_full_seq_len": full_len,
        }

    history = {
        k: {
            (s.sample_id, condition): observe(None, s, k, None, 3 + k, None)
            for s in representatives
            for condition in ("own", "drop")
        }
        for k in (1, 4)
    }
    if mismatch == "history":
        history[1][representatives[0].sample_id, "own"][
            "candidate_log_probabilities"
        ] = {"5": -0.10001}
    if mismatch == "prediction":
        history[1][representatives[0].sample_id, "own"]["predicted_digit"] = 6
    counter = 0

    def forward(*args, **kwargs):
        nonlocal counter
        counter += 1
        return SimpleNamespace(
            logits=torch.tensor([counter % 2 if mismatch == "logits" else 0.0])
        )

    monkeypatch.setattr(parity, "build_sender_cache", build)
    monkeypatch.setattr(parity, "observe", observe)
    model = SimpleNamespace(
        model_name="model", device="cpu", forward_next_token_batch=forward
    )
    report = parity.verify_prefix_parity(
        model, representatives, (None, None, None), history
    )
    assert len(report["checks"]) == 20
    assert report["passed"] == (mismatch in (None, "history"))


def trajectory_fixture():
    samples = generate_secret_digit_samples(100, 42)
    records = []
    for sample in samples:
        base = {
            "sample_id": sample.sample_id,
            "sample_key": sample.sample_key,
            "target_digit": sample.digit,
            "source_digit": sample.digit,
            "model": "Qwen/Qwen3-0.6B",
            "seed": 42,
            "error": None,
            "predicted_digit": 5,
        }
        records.append(base | {"condition": "drop"})
        for step in range(1, 21):
            records.append(
                base
                | {
                    "condition": "own",
                    "latent_steps": step,
                    "context_mode": "latent_only",
                    "cache_sequence_length": step,
                    "receiver_position_start": 34 + step,
                    "original_full_seq_len": 34 + step,
                    "retained_tail_start_position": 34,
                    "retained_original_position_end": 34 + step,
                    "predicted_digit": 4 if sample.digit < 3 else 5,
                }
            )
    probes = [
        {
            "model": "Qwen/Qwen3-0.6B",
            "latent_steps": "20",
            "representation": "latent_post_realign",
            "step": str(k),
            "run_id": "probe",
            "probe_accuracy": "0.19",
            "null_mean_accuracy": "0.11",
            "significance": "True",
            "fwer_p_value": "0.01",
        }
        for k in range(1, 21)
    ]
    return records, probes


def test_summary_pairs_canonical_receiver_samples_and_uses_probe_null_not_chance():
    records, probes = trajectory_fixture()
    rows = summarize_trajectory(records, probes, "Qwen/Qwen3-0.6B")
    assert len(rows) == 20
    assert all(r["probe_effect_pp"] == pytest.approx(8.0) for r in rows)
    assert all(r["receiver_changed_fraction"] == pytest.approx(0.3) for r in rows)


@pytest.mark.parametrize("corruption", ["duplicate", "identity", "position", "missing"])
def test_summary_rejects_incomplete_unpaired_or_mispositioned_cells(corruption):
    records, probes = trajectory_fixture()
    if corruption == "duplicate":
        records[22] = records[1].copy()
    elif corruption == "identity":
        records[1]["sample_key"] = "probe-template-sample"
    elif corruption == "position":
        records[1]["receiver_position_start"] += 20
    else:
        records.pop()
    with pytest.raises(ValueError):
        summarize_trajectory(records, probes, "Qwen/Qwen3-0.6B")


@pytest.mark.parametrize("corruption", ["smoke", "duplicate"])
def test_scatter_rejects_smoke_or_missing_model_step_cells(tmp_path, corruption):
    from src.run.receiver_trajectory_plots import plot_trajectory

    rows = [
        {
            "model": f"Qwen/Qwen3-{model}",
            "latent_step": str(step),
            "sample_count": "100",
        }
        for model in ("0.6B", "4B", "8B")
        for step in range(1, 21)
    ]
    if corruption == "smoke":
        rows[0]["sample_count"] = "10"
    else:
        rows[-1] = rows[0].copy()
    path = tmp_path / "trajectory.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        plot_trajectory(path, tmp_path / "plots")
    assert not (tmp_path / "plots").exists()


def test_scatter_accepts_complete_requested_schema_and_connects_step_order(
    tmp_path, monkeypatch
):
    from src.run import receiver_trajectory_plots as plots

    records, probes = trajectory_fixture()
    base = summarize_trajectory(records, probes, "Qwen/Qwen3-0.6B")
    rows = [
        row | {"model": f"Qwen/Qwen3-{size}"}
        for size in ("0.6B", "4B", "8B")
        for row in reversed(base)
    ]
    path = tmp_path / "trajectory.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = []
    monkeypatch.setattr(plots, "save", lambda fig, *args: figures.append(fig))
    plots.plot_trajectory(path, tmp_path / "plots")
    ax = figures[0].axes[0]
    connections = [line for line in ax.lines if len(line.get_xdata()) == 20]
    assert len(connections) == 3
    assert all(list(line.get_ydata()) == [30.0] * 20 for line in connections)
    plots.plt.close(figures[0])
