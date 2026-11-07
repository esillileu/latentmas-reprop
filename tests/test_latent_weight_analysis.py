"""Recorded states retain injection indexing and actual-start context."""

import pytest
import torch

from latentmas_reprop.application.common.statistics import (
    McNemarResult,
    paired_uncertainty,
)
from latentmas_reprop.application.latent_weight_analysis import (
    distribution_features,
    embedding_comparison,
    injected_vectors,
)


def test_paired_interval_handles_concordance_and_direction():
    zero = paired_uncertainty(McNemarResult(2, 0, 0, 0, 0, 0, 1))
    assert zero["accuracy_diff_ci_low"] < 0 < zero["accuracy_diff_ci_high"]
    positive = paired_uncertainty(McNemarResult(0, 40, 0, 0, 1, 40, 0))
    negative = paired_uncertainty(McNemarResult(0, 0, 40, 0, -1, 40, 0))
    assert positive["accuracy_diff_ci_low"] > 0
    assert positive["accuracy_diff_ci_low"] == pytest.approx(
        -negative["accuracy_diff_ci_high"]
    )
    assert (
        paired_uncertainty(McNemarResult(0, 0, 0, 0, 0, 0, 1))["mcnemar_exact_p"]
        is None
    )


def test_injections_align_previous_hidden_and_cast_after_normalizing():
    states = torch.tensor([[3.0, 4.0], [0.0, 0.0]], dtype=torch.float16)
    matrix = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    vectors = injected_vectors(states, matrix, torch.tensor(10.0))
    assert vectors.dtype == states.dtype
    assert torch.equal(
        vectors, torch.tensor([[8.0, 6.0], [0.0, 0.0]], dtype=torch.float16)
    )
    comparison = embedding_comparison(
        vectors[:1], torch.tensor([[8.0, 6.0], [-8.0, -6.0], [1.0, 0.0]]), chunk_size=1
    )
    assert comparison[0]["nearest_embedding_token_id"] == 0
    assert comparison[0]["nearest_embedding_cosine"] == pytest.approx(1.0)
    distribution = distribution_features(torch.tensor([1.0, 0.0]), torch.eye(2), [0, 1])
    assert distribution["think_probability"] > distribution["endthink_probability"]
    assert distribution["top_token_id"] == 0


def test_prompt_logits_are_available_without_any_readout(tmp_path):
    import csv
    import json

    from latentmas_reprop.application.intervention_continuation.identity import (
        _tensor_digest,
    )
    from src.run.latent_weight_analysis import analyze

    root = tmp_path / "run"
    sample = root / "samples/000000"
    sample.mkdir(parents=True)
    target = torch.tensor(1.0)
    metadata = {
        "model_name": "model",
        "task": "gsm8k",
        "latent_space_realign": False,
        "latent_step_start": 0,
        "latent_steps": 1,
        "state_identity": {
            "realignment_digest": _tensor_digest([("target_norm", target)])
        },
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    (sample / "sample.json").write_text(
        json.dumps({"index": 0, "thinking_token_ids": [0, 1]})
    )
    cache = tmp_path / "align"
    cache.mkdir()
    torch.save(
        {"identity": True, "target_norm": target}, cache / "model_identity_realign.pt"
    )
    for name in ("without_think", "with_think"):
        tensors = sample / "trajectories" / name
        tensors.mkdir(parents=True)
        torch.save({"hidden": torch.tensor([[0.0, 1.0]])}, tensors.with_suffix(".pt"))
        torch.save(torch.tensor([[1.0, 0.0]]), tensors / "prompt_hidden.pt")
    output = tmp_path / "report"
    analyze(root, torch.eye(2), torch.eye(2), output, cache)
    origins = list(csv.DictReader((output / "trajectory_origin_logits.csv").open()))
    starts = list(csv.DictReader((output / "readout_start_logits.csv").open()))
    assert len(origins) == 2
    assert all(
        row["context"] == "prompt_raw" and row["top_token_id"] == "0" for row in origins
    )
    assert all(row["status"] == "missing" for row in starts)
