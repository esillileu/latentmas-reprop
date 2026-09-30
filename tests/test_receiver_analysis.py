"""Saved-observation receiver diagnostics."""

import pytest

from latentmas_reprop.application.receiver_acquisition.analysis import (
    analyze_records,
    render_markdown,
    select_receiver_runs,
)


def _candidate(run_id, model="model", steps=4, samples=100, designated=False):
    return {
        "run_id": run_id,
        "model": model,
        "latent_steps": steps,
        "sample_count": samples,
        "condition_counts": {
            "drop": samples,
            "latent_only/own": samples,
            "latent_only/cross": samples,
        },
        "designated": designated,
    }


def test_selection_excludes_partial_runs_and_reports_coverage():
    selected = select_receiver_runs(
        [_candidate("partial", samples=10), _candidate("complete")], set()
    )
    assert [run["run_id"] for run in selected] == ["complete"]
    report = render_markdown([{**selected[0], "conditions": {}}])
    assert "| model | 4 | complete | 100 |" in report
    assert "partial" not in report


def test_selection_rejects_ambiguous_canonical_runs():
    candidates = [_candidate("first"), _candidate("second")]
    with pytest.raises(ValueError, match=r"Ambiguous canonical.*first.*second"):
        select_receiver_runs(candidates, set())
    assert [run["run_id"] for run in select_receiver_runs(candidates, {"second"})] == [
        "second"
    ]
    with pytest.raises(ValueError, match="Ambiguous designated"):
        select_receiver_runs(
            [
                _candidate("first", designated=True),
                _candidate("second", designated=True),
            ],
            set(),
        )


def test_selection_rejects_unknown_designation():
    with pytest.raises(ValueError, match="not found"):
        select_receiver_runs([_candidate("known")], {"missing"})


def test_source_digit_confusion_and_paired_probability_delta():
    records = []
    for index, (target, source, own_prediction, cross_prediction) in enumerate(
        [(0, 7, 1, 7), (1, 7, 7, 1), (7, 1, 1, 1)]
    ):
        for condition, mode, prediction in (
            ("drop", "none", 3),
            ("own", "latent_only", own_prediction),
            ("cross", "latent_only", cross_prediction),
        ):
            probabilities = {str(digit): 0.1 for digit in range(10)}
            if condition != "drop":
                probabilities[str(source)] = 0.08
            records.append(
                {
                    "sample_index": index,
                    "target_digit": target,
                    "source_digit": source if condition != "drop" else None,
                    "predicted_digit": prediction,
                    "candidate_probabilities": probabilities,
                    "condition": condition,
                    "context_mode": mode,
                    "error": None,
                }
            )
    result = analyze_records(records)
    own = result["latent_only/own"]
    assert own["prediction_distribution"]["1"]["count"] == 2
    assert own["prediction_distribution"]["7"]["fraction"] == pytest.approx(1 / 3)
    assert own["confusion_matrix"][7][1] == 1
    assert own["per_digit_accuracy"][7]["sample_count"] == 2
    assert own["per_digit_accuracy"][7]["correct_count"] == 1
    assert own["unique_predictions"] == 2
    assert own["top_prediction"] == 1
    assert own["top_fraction"] == pytest.approx(2 / 3)
    assert own["accuracy"] == pytest.approx(2 / 3)
    assert own["accuracy_delta_vs_drop"] == pytest.approx(2 / 3)
    assert own["probability_delta_vs_drop"] == pytest.approx(-0.02)
    assert result["drop"]["confusion_matrix"][0][3] == 1
