"""Integrity checks for finite-budget presentation inputs."""

import csv
import json

import pytest

from src.run.receiver_budget_plots import (
    BUDGETS,
    CONDITIONS,
    MODELS,
    STEPS,
    read_budget_curves,
)


@pytest.fixture
def compute(tmp_path):
    for size in MODELS:
        directory = tmp_path / f"Qwen_Qwen3-{size}"
        directory.mkdir()
        (directory / "source.json").write_text(
            json.dumps({"model": f"Qwen/Qwen3-{size}"})
        )
        rows = [
            {
                "upstream_steps": u,
                "handoff_condition": c[0],
                "receiver_budget": b,
                "accuracy": 0.5,
                "sample_count": 2,
            }
            for u in STEPS
            for c in CONDITIONS
            for b in BUDGETS
        ]
        with (directory / "budget_curves.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        effects = [
            {
                "upstream_steps": u,
                "receiver_budget": b,
                "comparison": "pairing_gain",
                "difference": 0,
                "ci_low": -0.25,
                "ci_high": 0.25,
            }
            for u in STEPS
            for b in BUDGETS
        ]
        (directory / "metrics.json").write_text(
            json.dumps({"sample_count": 2, "comparisons": effects})
        )
    return tmp_path


def test_complete_grid_and_saved_intervals(compute):
    results = read_budget_curves(compute, sample_count=2)
    assert set(results) == set(MODELS)
    assert results["8B"][1][10, 64]["ci_low"] == -0.25
    with pytest.raises(ValueError, match="sample_count"):
        read_budget_curves(compute, sample_count=20)


@pytest.mark.parametrize("fault", ["duplicate", "missing", "nan", "baseline"])
def test_invalid_curves_fail(compute, fault):
    path = compute / "Qwen_Qwen3-4B" / "budget_curves.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "missing":
        rows.pop()
    else:
        rows[-1]["accuracy"] = "NaN" if fault == "nan" else "0.9"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        read_budget_curves(compute)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "ci", "disagreement"])
def test_invalid_saved_effects_fail(compute, fault):
    path = compute / "Qwen_Qwen3-4B" / "metrics.json"
    metrics = json.loads(path.read_text())
    rows = metrics["comparisons"]
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "ci":
        rows[-1]["ci_low"] = float("nan")
    else:
        rows[-1]["difference"] = 0.1
    path.write_text(json.dumps(metrics))
    with pytest.raises(ValueError):
        read_budget_curves(compute)
