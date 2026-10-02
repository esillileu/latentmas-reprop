"""Saved-probability permutation tests and multiple-comparison families."""

import numpy as np
import pytest

from latentmas_reprop.application.receiver_acquisition.inference import (
    analyze_probability_records,
    correct_receiver_families,
    enrich_condition,
)


def records(probabilities):
    return [
        {
            "sample_index": i,
            "target_digit": i % 2,
            "source_digit": i % 2,
            "predicted_digit": 0,
            "candidate_probabilities": {str(d): float(p) for d, p in enumerate(row)},
        }
        for i, row in enumerate(probabilities)
    ]


def test_soft_dependency_can_vary_with_constant_argmax():
    probabilities = []
    for i in range(40):
        q = np.full(10, 0.01)
        q[0] = 0.6 if i % 2 == 0 else 0.5
        q[1] = 0.25 if i % 2 == 0 else 0.35
        probabilities.append(q)
    group = records(probabilities)
    samples, recovery, tests = analyze_probability_records(
        group, drop=False, permutations=199, seed=13
    )
    assert recovery["raw_p_value"] == 1
    assert tests["source_label_nll"]["raw_p_value"] == pytest.approx(0.005)
    assert tests["brier_score"]["raw_p_value"] == pytest.approx(0.005)
    assert (
        tests["source_label_nll"]["observed"] < tests["source_label_nll"]["null_mean"]
    )
    assert samples[1]["correct_digit_rank"] == 2
    assert samples[1]["top_2"] is True
    assert samples[1]["correct_digit_margin"] < 0
    assert (samples, recovery, tests) == analyze_probability_records(
        group, drop=False, permutations=199, seed=13
    )


def test_uninformative_probabilities_and_missing_values():
    group = records(np.full((20, 10), 0.01))
    samples, _, tests = analyze_probability_records(
        group, drop=False, permutations=19, seed=1
    )
    assert tests["source_label_nll"]["raw_p_value"] == 1
    assert tests["brier_score"]["raw_p_value"] == 1
    assert samples[0]["correct_digit_rank"] == 1
    group[0]["candidate_probabilities"] = {}
    values = {"per_digit_accuracy": [{} for _ in range(10)]}
    enrich_condition(values, group, drop=False, permutations=19, seed=1)
    assert values["probability_samples"][0]["correct_digit_rank"] is None
    assert values["soft_dependency"]["brier_score"]["sample_count"] == 19
    assert values["per_digit_accuracy"][0]["probability_sample_count"] == 9
    assert values["per_digit_accuracy"][2]["mean_source_label_nll"] is None


def test_holm_fwer_covers_all_runs_and_excludes_drop():
    runs = []
    for p in (0.01, 0.03, 0.2):
        runs.append(
            {
                "conditions": {
                    "latent_only/own": {
                        "dependency": {"permutation_p_value": p},
                        "exact_recovery": {"raw_p_value": p},
                        "soft_dependency": {
                            "source_label_nll": {"raw_p_value": p},
                            "brier_score": None,
                        },
                    },
                    "drop": {"dependency": {"permutation_p_value": 0.001}},
                }
            }
        )
    correct_receiver_families(runs)
    assert [
        r["conditions"]["latent_only/own"]["dependency"]["holm_p_value"] for r in runs
    ] == pytest.approx([0.03, 0.06, 0.2])
    assert runs[0]["conditions"]["latent_only/own"]["dependency"]["family_size"] == 3
    assert "holm_p_value" not in runs[0]["conditions"]["drop"]["dependency"]


def test_markdown_tables_keep_rows_with_headers_and_include_new_statistics():
    from latentmas_reprop.application.receiver_acquisition.analysis import (
        analyze_records,
    )
    from latentmas_reprop.application.receiver_acquisition.markdown import (
        render_markdown,
    )

    group = records(np.full((20, 10), 0.01))
    saved = []
    for row in group:
        for condition in ("drop", "own", "cross"):
            saved.append(
                {
                    **row,
                    "condition": condition,
                    "context_mode": "latent_only",
                    "error": None,
                }
            )
    conditions = analyze_records(saved, permutations=19, seed=3)
    report = render_markdown(
        [
            {
                "model": "test",
                "latent_steps": 4,
                "run_id": "raw",
                "sample_count": 20,
                "conditions": conditions,
            }
        ]
    )
    lines = report.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("|---"):
            assert lines[index - 1].startswith("|")
            if lines[index - 1].startswith(
                ("| Model | Steps | Condition", "| Source |")
            ):
                assert lines[index + 1].startswith("|")
            columns = lines[index - 1].count("|")
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].startswith("|"):
                assert lines[cursor].count("|") == columns
                cursor += 1
    assert "MI FWER p (Holm)" in report
    assert "NLL observed | NLL null mean" in report
    assert "permutation count: 19, seed: 3" in report
    assert "Mean P(correct) | Mean q(correct) | NLL | Brier" in report
