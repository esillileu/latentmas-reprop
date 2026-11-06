"""All-item PCA shares axes across benchmarks and balances sampling density."""

import json

import pytest
import torch

from latentmas_reprop.application.latent_global_pca import (
    analyze_global_pca,
    global_projection,
)
from src.run.latent_global_pca import write_global_report


def trajectory(task, sample, run, value, count=1):
    return {
        "benchmark": task,
        "sample_id": sample,
        "source_run_id": run,
        "alignment": False,
        "trajectory_type": "without_think",
        "steps": list(range(count)),
        "hidden": torch.tensor([[value, 0.0]] * count),
    }


def test_hierarchical_weights_balance_benchmarks_items_trajectories_and_steps():
    groups = [
        trajectory("gsm8k", 0, "old", 0.0, 20),
        trajectory("gsm8k", 0, "new", 2.0),
        trajectory("gsm8k", 1, "old", 10.0),
        trajectory("medqa", 0, "old", 100.0),
    ]
    basis, rows = global_projection(groups)
    # GSM8K item means: 1 and 10; benchmark means: 5.5 and 100.
    assert basis["mean"][0] == pytest.approx(52.75)
    assert len(rows) == 23
    assert {r["benchmark"] for r in rows} == {"gsm8k", "medqa"}
    assert {r["source_run_id"] for r in rows} == {"old", "new"}
    assert basis["explained_variance_ratio"].tolist() == [1.0, 0.0]
    # Repeating identical saved points changes neither mean nor covariance.
    groups[0] = trajectory("gsm8k", 0, "old", 0.0, 2)
    other, _ = global_projection(groups)
    assert torch.allclose(basis["mean"], other["mean"])
    assert torch.allclose(basis["components"], other["components"])


def save_source(root, task, run, model, value, revision="old"):
    directory = root / task / model / run
    sample = directory / "samples/000000"
    sample.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "task": task,
                "model_name": model,
                "latent_space_realign": False,
                "latent_steps": 1,
                "model_revision": revision,
            }
        )
    )
    (sample / "sample.json").write_text(json.dumps({"index": 0}))
    tensors = sample / "trajectories"
    tensors.mkdir()
    for kind in ("without_think", "with_think"):
        torch.save(
            {"hidden": torch.tensor([[value, 0.0]]), "steps": torch.tensor([1])},
            tensors / f"{kind}.pt",
        )
    return directory


def test_saved_runs_are_used_as_is_and_models_are_separate(tmp_path):
    save_source(tmp_path, "gsm8k", "historical", "model", 0.0)
    save_source(tmp_path, "gsm8k", "current", "model", 10.0, revision="new")
    save_source(tmp_path, "medqa", "saved", "model", 100.0)
    save_source(tmp_path, "medqa", "saved", "other", 200.0)
    results = analyze_global_pca(tmp_path)
    assert [r["model"] for r in results] == ["model", "other"]
    first = results[0]
    assert len(first["coordinates"]) == 6
    assert first["basis"]["mean"][0] == pytest.approx(52.5)
    assert first["status"] == "partial"  # Prompt step 0 was not saved.
    output = tmp_path / "report"
    write_global_report(first, output)
    saved = torch.load(output / "global_pca.pt", weights_only=True)
    assert torch.equal(saved["mean"], first["basis"]["mean"])
    assert (output / "global_pca.png").exists()
    assert (output / "global_pca_coordinates.csv").exists()
    assert "No freshness checks" in (output / "summary.md").read_text()
    assert len(analyze_global_pca(tmp_path, "other")) == 1
    assert analyze_global_pca(tmp_path, "not present") == []
