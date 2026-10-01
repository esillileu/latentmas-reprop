"""Saved-observation receiver diagnostics."""

import pytest

from latentmas_reprop.application.receiver_acquisition.analysis import (
    analyze_records,
    select_receiver_runs,
)
from latentmas_reprop.application.receiver_acquisition.markdown import render_markdown


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


def test_cross_matching_missing_probabilities_and_dependency():
    records = []
    for index in range(10):
        for condition, prediction in (("drop", 0), ("cross", index)):
            records.append(
                {
                    "sample_index": index,
                    "target_digit": 0,
                    "source_digit": index,
                    "predicted_digit": prediction,
                    "candidate_probabilities": {},
                    "condition": condition,
                    "context_mode": "latent_only",
                    "error": None,
                }
            )
    result = analyze_records(records, permutations=99, seed=7)
    cross = result["latent_only/cross"]
    assert cross["accuracy"] == cross["balanced_accuracy"] == 1
    assert cross["matched_drop_accuracy"] == pytest.approx(0.1)
    assert cross["correct_class_probability"] is None
    assert cross["digit_probability_mass"] is None
    assert cross["dependency"]["normalized_mi"] == pytest.approx(1)
    assert len(cross["dependency"]["permutation_statistics"]) == 99
    assert cross["dependency"]["permutation_p_value"] == pytest.approx(1.0)
    assert result == analyze_records(records, permutations=99, seed=7)


def test_probability_mass_normalization_and_pairing():
    records = []
    for condition, mass in (("drop", 0.2), ("own", 0.4)):
        records.append(
            {
                "sample_index": 0,
                "target_digit": 0,
                "source_digit": 1,
                "predicted_digit": 1,
                "condition": condition,
                "context_mode": "latent_only",
                "error": None,
                "candidate_probabilities": {str(d): mass / 10 for d in range(10)},
            }
        )
    own = analyze_records(records, permutations=9)["latent_only/own"]
    assert own["digit_probability_mass"] == pytest.approx(0.4)
    assert own["digit_probability_mass_delta_vs_drop"] == pytest.approx(0.2)
    assert own["source_label_nll"] == pytest.approx(2.302585092994046)
    assert own["brier_score"] == pytest.approx(0.9)
    assert own["js_divergence_vs_drop"] == pytest.approx(0)
    assert own["dependency"]["normalized_mi"] is None


def test_sender_statistics_use_saved_permutation_family():
    from latentmas_reprop.application.receiver_acquisition.report import sender_cells

    specs = [
        {"representation": "hidden_pre_realign", "step": 1},
        {"representation": "latent_post_realign", "step": 1},
    ]
    results = {"cells": [{**s, "observed_oof_accuracy": 0.6} for s in specs]}
    stats = {
        "cell_order": specs,
        "permutation_accuracies": [[0.1, 0.7], [0.2, 0.3], [0.6, 0.4]],
    }
    cells = sender_cells(results, stats)
    assert cells[0]["null_mean_accuracy"] == pytest.approx(0.3)
    assert cells[0]["observed_minus_null"] == pytest.approx(0.3)
    assert cells[0]["raw_p_value"] == pytest.approx(0.5)
    assert cells[0]["fwer_p_value"] == pytest.approx(0.75)


def test_export_tables_preserves_statistics_and_corrects_family(tmp_path):
    import json

    from latentmas_reprop.application.receiver_acquisition.report import (
        export_tables,
        sender_cells,
    )

    spec = {"representation": "hidden_pre_realign", "step": 1}
    probes = sender_cells(
        {"cells": [{**spec, "observed_oof_accuracy": 0.5}]},
        {"cell_order": [spec], "permutation_accuracies": [[0.1], [0.6]]},
    )
    conditions = analyze_records(
        [
            {
                "sample_index": i,
                "target_digit": i % 2,
                "source_digit": i % 2,
                "predicted_digit": i % 2,
                "candidate_probabilities": {},
                "condition": condition,
                "context_mode": "latent_only",
                "error": None,
            }
            for i in range(20)
            for condition in ("drop", "own", "cross")
        ],
        permutations=19,
    )
    run = {
        "model": "test",
        "latent_steps": 1,
        "run_id": "saved",
        "conditions": conditions,
    }
    export_tables(tmp_path, [run], probes)
    dependencies = json.loads((tmp_path / "receiver_dependencies.json").read_text())
    assert len(dependencies) == 2
    for row in dependencies:
        assert row["holm_p_value"] >= row["permutation_p_value"]
        assert len(row["permutation_statistics"]) == 19
    assert len(json.loads((tmp_path / "receiver_per_digit.json").read_text())) == 30
    assert (
        json.loads((tmp_path / "sender_probe_cells.json").read_text())[0][
            "significance"
        ]
        is False
    )
