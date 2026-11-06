"""Common means/axes preserve condition offsets without cross-item pooling."""

import json

import pytest
import torch

from latentmas_reprop.application.latent_common_pca import (
    analyze_common_pca,
    common_projection,
)
from latentmas_reprop.application.latent_state_features import fit_pca
from src.run.latent_common_pca_report import write_common_pca_report


def trajectory(alignment, hidden, name="without_think"):
    return {
        "source_run_id": str(alignment),
        "sample_id": 0,
        "alignment": alignment,
        "trajectory_type": name,
        "steps": list(range(len(hidden))),
        "hidden": torch.tensor(hidden, dtype=torch.float32),
    }


def test_common_axes_keep_between_condition_offset_and_balance_unequal_lengths():
    groups = [trajectory(False, [[0.0, 0.0]]), trajectory(True, [[10.0, 0.0]] * 3)]
    basis, rows = common_projection(groups)
    assert torch.allclose(basis["mean"], torch.tensor([5.0, 0.0]))
    assert basis["explained_variance_ratio"].tolist() == [1.0, 0.0]
    assert rows[0]["common_pca_1"] == pytest.approx(-5)
    assert rows[-1]["common_pca_1"] == pytest.approx(5)
    for group in groups:
        projected = (group["hidden"] - basis["mean"]) @ basis["components"].T
        selected = [r for r in rows if r["alignment"] == group["alignment"]]
        assert torch.allclose(
            projected[:, 0], torch.tensor([r["common_pca_1"] for r in selected])
        )
    with pytest.raises(ValueError, match="Duplicate"):
        common_projection([groups[0], groups[0]])


def test_individual_centering_would_erase_the_offset_and_constant_fit_is_defined():
    off = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    on = off + torch.tensor([10.0, 0.0])
    assert torch.allclose(off - fit_pca(off)["mean"], on - fit_pca(on)["mean"])
    basis, _ = common_projection(
        [trajectory(False, off.tolist()), trajectory(True, on.tolist())]
    )
    projected_distance = ((on.mean(0) - off.mean(0)) @ basis["components"].T).norm()
    assert projected_distance == pytest.approx(10)
    constant = fit_pca(torch.ones(3, 2))
    assert constant["explained_variance_ratio"].tolist() == [0.0, 0.0]


def source(tmp_path, task, name, alignment, *, sample_id=0, seed=42):
    root = tmp_path / task / "Model" / name
    sample = root / f"samples/{sample_id:06d}"
    sample.mkdir(parents=True)
    metadata = {
        "task": task,
        "model_name": "Model",
        "latent_space_realign": alignment,
        "latent_steps": 2,
        "seed": seed,
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    (sample / "sample.json").write_text(
        json.dumps(
            {
                "index": sample_id,
                "question": f"q{sample_id}",
                "gold": "c",
                "prompt_token_ids": [1, 2],
            }
        )
    )
    for kind in ("without_think", "with_think"):
        tensors = sample / "trajectories"
        (tensors / kind).mkdir(parents=True)
        offset = 10.0 if alignment else 0.0
        torch.save(
            {
                "steps": torch.tensor([1, 2]),
                "hidden": torch.tensor([[1.0 + offset, 0.0], [2.0 + offset, 1.0]]),
            },
            tensors / f"{kind}.pt",
        )
        torch.save(torch.tensor([[offset, 0.0]]), tensors / kind / "prompt_hidden.pt")
    return root, metadata


@pytest.mark.parametrize("task", ["gsm8k", "medqa"])
def test_same_per_item_workflow_for_both_tasks_with_missing_conditions(tmp_path, task):
    root, metadata = source(tmp_path, task, "off", False)
    results = analyze_common_pca(root, metadata)
    assert results[0]["status"] == "partial"
    assert len(results[0]["coordinates"]) == 6
    assert [g["status"] for g in results[0]["groups"]] == [
        "complete",
        "complete",
        "missing",
        "missing",
    ]
    other, _ = source(tmp_path, task, "on", True)
    result = analyze_common_pca(root, metadata)[0]
    assert result["status"] == "complete"
    reverse = analyze_common_pca(
        other, json.loads((other / "metadata.json").read_text())
    )[0]
    assert torch.allclose(reverse["basis"]["mean"], result["basis"]["mean"])
    assert torch.allclose(reverse["basis"]["components"], result["basis"]["components"])
    assert (
        len(result["coordinates"]) == 12
    )  # Four shared trajectories, not eight A-D readouts.
    source(tmp_path, task, "off", False, sample_id=145)
    source(tmp_path, task, "on", True, sample_id=145)
    results = analyze_common_pca(root, metadata)
    assert [r["sample_id"] for r in results] == [0, 145]
    assert all(len(r["coordinates"]) == 12 for r in results)
    # One item has one basis even when another item exists in the source.
    assert torch.equal(results[0]["basis"]["mean"], result["basis"]["mean"])
    (other / "samples/000000/trajectories/with_think/prompt_hidden.pt").unlink()
    partial = analyze_common_pca(root, metadata)[0]
    assert partial["status"] == "partial" and len(partial["coordinates"]) == 11


def test_incompatible_or_ambiguous_sources_are_not_silently_pooled(tmp_path):
    root, metadata = source(tmp_path, "gsm8k", "off", False)
    other, _ = source(tmp_path, "medqa", "on", True)
    result = analyze_common_pca(root, metadata, other)[0]
    assert result["status"] == "partial"
    assert "Incompatible" in result["notes"][0]
    source(tmp_path, "gsm8k", "on1", True)
    source(tmp_path, "gsm8k", "on2", True)
    result = analyze_common_pca(root, metadata)[0]
    assert "Multiple" in result["notes"][0]
    assert len(result["coordinates"]) == 6
    explicit = analyze_common_pca(root, metadata, root.parent / "on1")[0]
    assert explicit["status"] == "complete"


def test_different_prompt_or_bad_tensor_stays_unavailable_and_report_preserves_basis(
    tmp_path,
):
    root, metadata = source(tmp_path, "gsm8k", "off", False)
    other, _ = source(tmp_path, "gsm8k", "on", True)
    sample = other / "samples/000000/sample.json"
    record = json.loads(sample.read_text())
    record["question"] = "different"
    sample.write_text(json.dumps(record))
    results = analyze_common_pca(root, metadata)
    assert all(
        g["status"] == "incompatible" for g in results[0]["groups"] if g["alignment"]
    )
    output = tmp_path / "report"
    output.mkdir()
    write_common_pca_report(results, output)
    saved = torch.load(output / "common_pca_000000.pt", weights_only=True)
    assert torch.equal(saved["mean"], results[0]["basis"]["mean"])
    assert (output / "common_pca_000000.png").exists()
    assert "Total explained variance" in (output / "common_pca_summary.md").read_text()
    status = json.loads((output / "common_pca_status.json").read_text())
    assert "hidden" not in status[0]["groups"][0]
    # A corrupt hidden artifact can still contribute an independently saved prompt state.
    (root / "samples/000000/trajectories/without_think.pt").write_bytes(b"corrupt")
    partial = analyze_common_pca(root, metadata)[0]
    assert partial["groups"][0]["status"] == "partial"
    assert partial["groups"][0]["steps"] == [0]
