"""Shared-state indexing, partial components and descriptive readout joins."""

import json

import pytest
import torch

from latentmas_reprop.application.latent_trajectory_analysis import (
    analyze_trajectories,
    join_readout_features,
    trajectory_metrics,
)


def data():
    return {
        "steps": torch.arange(1, 4),
        "hidden": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "logits": torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1000.0, -1000.0]]
        ),
    }


def test_prompt_origin_indexing_and_logit_reference_are_distinct():
    rows = trajectory_metrics(data(), torch.tensor([[90.0, 99.0], [0.0, 0.0]]), [0, 1])
    assert [r["step"] for r in rows] == [0, 1, 2, 3]
    assert rows[0]["l2_norm"] == 0 and rows[1]["distance_0"] == 1
    assert rows[1]["adjacent_l2"] == 1 and rows[2]["direction_cosine"] == 1
    assert rows[0]["logits_status"] == "missing"
    assert rows[1]["js_divergence"] is None
    assert rows[2]["js_divergence"] == pytest.approx(0, abs=1e-6)
    assert 0 < rows[3]["js_divergence"] <= 0.693148
    assert rows[3]["logit_reference_step"] == 1
    assert rows[3]["think_logit"] == 0 and rows[3]["endthink_logit"] == 1000
    assert rows[3]["endthink_probability"] == 1
    assert rows[3]["effective_rank"] == pytest.approx(1)
    assert rows[3]["numerical_rank"] == 1


def test_components_survive_each_others_missing_or_invalid_data():
    invalid = data()
    invalid["hidden"][0, 0] = float("nan")
    rows = trajectory_metrics(invalid)
    assert rows[1]["hidden_status"] == "invalid"
    assert rows[1]["logits_status"] == "complete"
    assert "entropy" in rows[1]
    only_hidden = data()
    del only_hidden["logits"]
    rows = trajectory_metrics(only_hidden)
    assert rows[1]["hidden_status"] == "complete"
    assert rows[1]["logits_status"] == "missing"
    assert rows[1]["distance_0"] is None
    rows = trajectory_metrics({}, torch.ones(2, 2), latent_steps=3)
    assert rows[0]["hidden_status"] == "complete"
    assert rows[1]["hidden_status"] == "missing"


def test_invalid_indices_do_not_silently_shift_steps_and_rank_zero_is_defined():
    invalid = data()
    invalid["steps"] = torch.tensor([0, 1, 2])
    assert trajectory_metrics(invalid)[1]["logits_status"] == "invalid"
    constant = data()
    constant["hidden"] = torch.ones(3, 2)
    rows = trajectory_metrics(constant)
    assert rows[1]["effective_rank"] == 0
    assert rows[1]["pca_explained_variance"] == [0, 0]
    invalid["hidden"] = "bad artifact"
    assert trajectory_metrics(invalid)[1]["hidden_status"] == "invalid"


def test_shared_state_join_keeps_condition_specific_outcomes_and_prefill_context():
    shared = [
        {
            **r,
            "source_run_id": "run",
            "sample_id": 0,
            "trajectory_type": "without_think",
        }
        for r in trajectory_metrics(data())
    ]
    readouts = [
        {
            "source_run_id": "run",
            "sample_id": 0,
            "trajectory_type": "without_think",
            "step": 2,
            "condition": condition,
            "status": "complete",
            "evaluator_correct": correct,
        }
        for condition, correct in (("a", True), ("c", False))
    ]
    joined = join_readout_features(readouts, shared)
    assert len(shared) == 4 and len(joined) == 2
    assert joined[0]["entropy"] == joined[1]["entropy"]
    assert joined[0]["evaluator_correct"] and not joined[1]["evaluator_correct"]
    assert joined[0]["readout_start_logits_status"] == "complete"
    assert "after_closing_prefill" in joined[1]["readout_start_logits_status"]


def test_files_missing_or_corrupt_are_reported_without_loading_kv(tmp_path):
    root = tmp_path / "source"
    sample = root / "samples/000000"
    sample.mkdir(parents=True)
    (sample / "sample.json").write_text(
        json.dumps({"index": 0, "thinking_token_ids": [0, 1]})
    )
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "task": "test",
                "model_name": "model",
                "latent_space_realign": False,
                "latent_steps": 3,
            }
        )
    )
    tensors = sample / "trajectories"
    tensors.mkdir()
    torch.save(data(), tensors / "without_think.pt")
    (tensors / "with_think.pt").write_bytes(b"corrupt")
    (tensors / "kv.pt").write_bytes(b"must never be read")
    statuses, rows = analyze_trajectories(root, tmp_path / "output")
    assert statuses[0]["status"] == "complete"
    assert statuses[1]["status"] == "invalid"
    assert len(rows) == 8
    assert (tensors / "without_think.pt").exists()
    assert not list(tensors.glob("*.analysis.json"))
