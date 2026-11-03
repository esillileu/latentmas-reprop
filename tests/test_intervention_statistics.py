"""Tests for condition outcome extraction and McNemar statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentmas_reprop.application.common.statistics import McNemarResult, mcnemar
from latentmas_reprop.application.intervention_continuation.outcomes import (
    ConditionOutcome,
)


def test_condition_outcome_properties() -> None:
    outcome = ConditionOutcome(
        condition="c1",
        sample_ids=[0, 1, 2, 3],
        correct=[True, False, True, True],
    )
    assert outcome.condition == "c1"
    assert outcome.n_samples == 4
    assert outcome.n_correct == 3
    assert outcome.accuracy == 0.75


def test_condition_outcome_empty() -> None:
    outcome = ConditionOutcome(
        condition="empty",
        sample_ids=[],
        correct=[],
    )
    assert outcome.n_samples == 0
    assert outcome.n_correct == 0
    assert outcome.accuracy == 0.0


def test_from_records_sorting_and_validation() -> None:
    records = [
        {"sample_index": 2, "condition": "rqle", "status": "complete", "correct": False, "gold": "10"},
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": True, "gold": "5"},
        {"sample_index": 1, "condition": "rqle", "status": "complete", "correct": True, "gold": "7"},
    ]
    outcome = ConditionOutcome.from_records(records)
    assert outcome.sample_ids == [0, 1, 2]
    assert outcome.correct == [True, True, False]
    assert outcome.golds == ["5", "7", "10"]
    assert outcome.n_correct == 2
    assert outcome.n_samples == 3


def test_from_records_duplicate_sample_id_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": True},
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": False},
    ]
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        ConditionOutcome.from_records(records)


def test_from_records_incomplete_sample_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": True},
        {"sample_index": 1, "condition": "rqle", "status": "running", "correct": False},
    ]
    with pytest.raises(ValueError, match="Incomplete sample"):
        ConditionOutcome.from_records(records)


def test_from_records_missing_sample_id_in_sequence_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": True},
        {"sample_index": 2, "condition": "rqle", "status": "complete", "correct": False},
    ]
    with pytest.raises(ValueError, match="Missing sample IDs"):
        ConditionOutcome.from_records(records)


def test_from_records_summary_mismatch_raises(tmp_path: Path) -> None:
    records = [
        {"sample_index": 0, "condition": "rqle", "status": "complete", "correct": True},
        {"sample_index": 1, "condition": "rqle", "status": "complete", "correct": False},
    ]
    summary_data = {
        "condition_statistics": {
            "rqle": {
                "accuracy": 0.8,
                "correct": 2,
            }
        }
    }
    with open(tmp_path / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f)

    with pytest.raises(ValueError, match="Accuracy mismatch"):
        ConditionOutcome.from_records(records, run_dir=tmp_path)


def test_mcnemar_identical_conditions() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[0, 1, 2, 3],
        correct=[True, False, True, False],
        dataset="gsm8k",
        split="test",
        golds=["1", "2", "3", "4"],
    )
    res = mcnemar(outcome_a, outcome_a)
    assert isinstance(res, McNemarResult)
    assert res.n11 == 2
    assert res.n00 == 2
    assert res.n10 == 0
    assert res.n01 == 0
    assert res.discordant_count == 0
    assert res.accuracy_diff == 0.0
    assert res.p_value == 1.0
    assert res.statistic == 0.0
    assert res.contingency_table == {"n11": 2, "n10": 0, "n01": 0, "n00": 2}


def test_mcnemar_synthetic_vectors() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[0, 1, 2, 3],
        correct=[True, True, False, False],
    )
    outcome_b = ConditionOutcome(
        condition="cond_b",
        sample_ids=[0, 1, 2, 3],
        correct=[True, False, True, False],
    )
    res = mcnemar(outcome_a, outcome_b)
    # sid 0: a=1, b=1 -> n11 (1)
    # sid 1: a=1, b=0 -> n10 (1)
    # sid 2: a=0, b=1 -> n01 (1)
    # sid 3: a=0, b=0 -> n00 (1)
    assert res.n11 == 1
    assert res.n10 == 1
    assert res.n01 == 1
    assert res.n00 == 1
    assert res.discordant_count == 2
    assert res.accuracy_diff == 0.0
    assert res.p_value == 1.0


def test_mcnemar_unsorted_sample_ids_are_aligned() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[2, 0, 1],
        correct=[False, True, True],  # 0: True, 1: True, 2: False
    )
    outcome_b = ConditionOutcome(
        condition="cond_b",
        sample_ids=[1, 2, 0],
        correct=[False, True, True],  # 0: True, 1: False, 2: True
    )
    res = mcnemar(outcome_a, outcome_b)
    # sid 0: a=1, b=1 -> n11 (1)
    # sid 1: a=1, b=0 -> n10 (1)
    # sid 2: a=0, b=1 -> n01 (1)
    assert res.n11 == 1
    assert res.n10 == 1
    assert res.n01 == 1
    assert res.n00 == 0


def test_mcnemar_sample_mismatch_raises() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[0, 1],
        correct=[True, False],
    )
    outcome_b = ConditionOutcome(
        condition="cond_b",
        sample_ids=[1, 2],
        correct=[False, True],
    )
    with pytest.raises(ValueError, match="Sample ID sets do not match"):
        mcnemar(outcome_a, outcome_b)


def test_mcnemar_gold_mismatch_raises() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[0, 1],
        correct=[True, False],
        golds=["42", "100"],
    )
    outcome_b = ConditionOutcome(
        condition="cond_b",
        sample_ids=[0, 1],
        correct=[True, False],
        golds=["42", "999"],
    )
    with pytest.raises(ValueError, match="Gold answer mismatch"):
        mcnemar(outcome_a, outcome_b)


def test_mcnemar_dataset_and_split_mismatch_raises() -> None:
    outcome_a = ConditionOutcome(
        condition="cond_a",
        sample_ids=[0],
        correct=[True],
        dataset="gsm8k",
        split="test",
    )
    outcome_b = ConditionOutcome(
        condition="cond_b",
        sample_ids=[0],
        correct=[True],
        dataset="math",
        split="test",
    )
    with pytest.raises(ValueError, match="Dataset mismatch"):
        mcnemar(outcome_a, outcome_b)

    outcome_c = ConditionOutcome(
        condition="cond_c",
        sample_ids=[0],
        correct=[True],
        dataset="gsm8k",
        split="train",
    )
    with pytest.raises(ValueError, match="Split mismatch"):
        mcnemar(outcome_a, outcome_c)


def test_existing_experiment_runs_agreement() -> None:
    """Verify against actual intervention experiment run artifacts in storage."""
    base_dir = Path("storage/experiments/intervention_continuation/gsm8k/Qwen3-8B/60cd515c728184529f4018c6e24121dfc1a30bc19c02ff0111c18eecc7453e8a/runs")
    run_a = base_dir / "0b7f21996f6ab2f45fdee9e7fa580d4f8697ec0717569d892c94d5ce0f36818a"
    run_b = base_dir / "11d9bf7de214773efe3e3ad7f86311138eedb47f4da21410b4865f05562bd217"

    if not run_a.exists() or not run_b.exists():
        pytest.skip("Existing experiment runs not found on disk")

    outcome_a = ConditionOutcome.from_run(run_a)
    outcome_b = ConditionOutcome.from_run(run_b)

    assert outcome_a.condition == "rqrqe"
    assert outcome_a.n_samples == 1319
    assert outcome_a.n_correct == 1198

    assert outcome_b.condition == "rqle"
    assert outcome_b.n_samples == 1319
    assert outcome_b.n_correct == 1195

    res = mcnemar(outcome_a, outcome_b)
    assert res.n11 == 1144
    assert res.n10 == 54
    assert res.n01 == 51
    assert res.n00 == 70
    assert res.discordant_count == 105
    assert abs(res.p_value - 0.8453703188825519) < 1e-6
