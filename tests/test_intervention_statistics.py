"""Tests for condition outcome extraction and McNemar statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
        {
            "sample_index": 2,
            "condition": "c",
            "status": "complete",
            "correct": False,
            "gold": "10",
            "task": "t",
            "split": "s",
        },
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "gold": "5",
            "task": "t",
            "split": "s",
        },
        {
            "sample_index": 1,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "gold": "7",
            "task": "t",
            "split": "s",
        },
    ]
    outcome = ConditionOutcome.from_records(records)
    assert outcome.sample_ids == [0, 1, 2]
    assert outcome.correct == [True, True, False]
    assert outcome.golds == ["5", "7", "10"]
    assert outcome.dataset == "t"
    assert outcome.split == "s"
    assert outcome.n_correct == 2
    assert outcome.n_samples == 3


def test_from_records_duplicate_sample_id_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "c", "status": "complete", "correct": True},
        {"sample_index": 0, "condition": "c", "status": "complete", "correct": False},
    ]
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        ConditionOutcome.from_records(records)


def test_from_records_incomplete_sample_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "c", "status": "complete", "correct": True},
        {"sample_index": 1, "condition": "c", "status": "running", "correct": False},
    ]
    with pytest.raises(ValueError, match="Incomplete sample"):
        ConditionOutcome.from_records(records)


def test_from_records_missing_sample_id_in_sequence_raises() -> None:
    records = [
        {"sample_index": 0, "condition": "c", "status": "complete", "correct": True},
        {"sample_index": 2, "condition": "c", "status": "complete", "correct": False},
    ]
    with pytest.raises(ValueError, match="Missing sample IDs"):
        ConditionOutcome.from_records(records)


def test_from_records_internal_metadata_inconsistency_raises() -> None:
    r_dataset = [
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "dataset": "d1",
        },
        {
            "sample_index": 1,
            "condition": "c",
            "status": "complete",
            "correct": False,
            "dataset": "d2",
        },
    ]
    with pytest.raises(ValueError, match="Inconsistent datasets"):
        ConditionOutcome.from_records(r_dataset)

    r_split = [
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "split": "s1",
        },
        {
            "sample_index": 1,
            "condition": "c",
            "status": "complete",
            "correct": False,
            "split": "s2",
        },
    ]
    with pytest.raises(ValueError, match="Inconsistent splits"):
        ConditionOutcome.from_records(r_split)


def test_from_records_summary_mismatches_raise(tmp_path: Path) -> None:
    records = [
        {"sample_index": 0, "condition": "c", "status": "complete", "correct": True},
        {"sample_index": 1, "condition": "c", "status": "complete", "correct": False},
    ]
    summary_path = tmp_path / "summary.json"

    # Accuracy mismatch
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "condition_statistics": {
                    "c": {"accuracy": 0.8, "correct": 2, "completed": 2}
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Accuracy mismatch"):
        ConditionOutcome.from_records(records, run_dir=tmp_path)

    # Sample count mismatch
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": 3,
                "condition_statistics": {
                    "c": {"accuracy": 0.5, "correct": 1, "completed": 2}
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Sample count mismatch"):
        ConditionOutcome.from_records(records, run_dir=tmp_path)

    # Completed count mismatch
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "condition_statistics": {
                    "c": {"accuracy": 0.5, "correct": 1, "completed": 1}
                },
            }
        )
    )
    with pytest.raises(ValueError, match="Completed count mismatch"):
        ConditionOutcome.from_records(records, run_dir=tmp_path)


def test_from_run_reconstructs_from_fixture(tmp_path: Path) -> None:
    """Validate from_run with state.json, summary.json, and samples.jsonl fixture."""
    state_dir = tmp_path / "state"
    run_dir = state_dir / "runs" / "r1"
    run_dir.mkdir(parents=True)

    (state_dir / "state.json").write_text(
        json.dumps({"identity": {"task": "d_fix", "split": "s_fix"}})
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "sample_count": 3,
                "condition_statistics": {
                    "c": {"completed": 3, "correct": 2, "accuracy": 2 / 3}
                },
            }
        )
    )
    records = [
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "gold": "10",
        },
        {
            "sample_index": 1,
            "condition": "c",
            "status": "complete",
            "correct": False,
            "gold": "20",
        },
        {
            "sample_index": 2,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "gold": "30",
        },
    ]
    (run_dir / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in records))

    outcome = ConditionOutcome.from_run(run_dir)
    assert outcome.condition == "c"
    assert outcome.dataset == "d_fix"
    assert outcome.split == "s_fix"
    assert outcome.n_samples == 3
    assert outcome.n_correct == 2
    assert outcome.accuracy == pytest.approx(2 / 3)
    assert outcome.golds == ["10", "20", "30"]


def test_from_records_task_and_split_conflict_with_state_raises() -> None:
    records_task_conflict = [
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "task": "gsm8k",
        },
    ]
    with pytest.raises(ValueError, match="Task/dataset conflict"):
        ConditionOutcome.from_records(records_task_conflict, default_dataset="math")

    records_split_conflict = [
        {
            "sample_index": 0,
            "condition": "c",
            "status": "complete",
            "correct": True,
            "split": "test",
        },
    ]
    with pytest.raises(ValueError, match="Split conflict"):
        ConditionOutcome.from_records(records_split_conflict, default_split="train")
