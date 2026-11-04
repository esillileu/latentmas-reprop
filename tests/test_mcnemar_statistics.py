"""Tests for condition outcome extraction and McNemar statistical tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latentmas_reprop.application.common.statistics import McNemarResult, mcnemar
from latentmas_reprop.application.intervention_continuation.outcomes import (
    ConditionOutcome,
)


def test_mcnemar_identical_conditions() -> None:
    outcome = ConditionOutcome(
        condition="c",
        sample_ids=[0, 1, 2, 3],
        correct=[True, False, True, False],
        dataset="d",
        split="s",
        golds=["1", "2", "3", "4"],
    )
    res = mcnemar(outcome, outcome)
    assert isinstance(res, McNemarResult)
    assert res.n11 == 2 and res.n00 == 2 and res.n10 == 0 and res.n01 == 0
    assert res.discordant_count == 0
    assert res.accuracy_diff == 0.0
    assert res.p_value == 1.0
    assert res.statistic == 0.0
    assert res.contingency_table == {"n11": 2, "n10": 0, "n01": 0, "n00": 2}


def test_mcnemar_synthetic_vectors() -> None:
    outcome_a = ConditionOutcome(
        condition="ca", sample_ids=[0, 1, 2, 3], correct=[True, True, False, False]
    )
    outcome_b = ConditionOutcome(
        condition="cb", sample_ids=[0, 1, 2, 3], correct=[True, False, True, False]
    )
    res = mcnemar(outcome_a, outcome_b)
    assert res.n11 == 1 and res.n10 == 1 and res.n01 == 1 and res.n00 == 1
    assert res.discordant_count == 2
    assert res.accuracy_diff == 0.0
    assert res.p_value == 1.0


def test_mcnemar_unsorted_sample_ids_are_aligned() -> None:
    outcome_a = ConditionOutcome(
        condition="ca", sample_ids=[2, 0, 1], correct=[False, True, True]
    )
    outcome_b = ConditionOutcome(
        condition="cb", sample_ids=[1, 2, 0], correct=[False, True, True]
    )
    res = mcnemar(outcome_a, outcome_b)
    assert res.n11 == 1 and res.n10 == 1 and res.n01 == 1 and res.n00 == 0


def test_mcnemar_sample_mismatch_raises() -> None:
    oa = ConditionOutcome(condition="ca", sample_ids=[0, 1], correct=[True, False])
    ob = ConditionOutcome(condition="cb", sample_ids=[1, 2], correct=[False, True])
    with pytest.raises(ValueError, match="Sample ID sets do not match"):
        mcnemar(oa, ob)


def test_mcnemar_gold_mismatch_raises() -> None:
    oa = ConditionOutcome(
        condition="ca", sample_ids=[0, 1], correct=[True, False], golds=["a", "b"]
    )
    ob = ConditionOutcome(
        condition="cb", sample_ids=[0, 1], correct=[True, False], golds=["a", "diff"]
    )
    with pytest.raises(ValueError, match="Gold answer mismatch"):
        mcnemar(oa, ob)


def test_mcnemar_dataset_and_split_mismatch_raises() -> None:
    oa = ConditionOutcome(
        condition="ca", sample_ids=[0], correct=[True], dataset="d1", split="s"
    )
    ob = ConditionOutcome(
        condition="cb", sample_ids=[0], correct=[True], dataset="d2", split="s"
    )
    with pytest.raises(ValueError, match="Dataset mismatch"):
        mcnemar(oa, ob)

    oc = ConditionOutcome(
        condition="cc", sample_ids=[0], correct=[True], dataset="d1", split="other"
    )
    with pytest.raises(ValueError, match="Split mismatch"):
        mcnemar(oa, oc)


def test_mcnemar_two_fixture_runs_comparison(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    run_a = state_dir / "runs" / "ra"
    run_b = state_dir / "runs" / "rb"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    (state_dir / "state.json").write_text(
        json.dumps({"identity": {"task": "d", "split": "s"}})
    )

    recs_a = [
        {
            "sample_index": 0,
            "condition": "ca",
            "status": "complete",
            "correct": True,
            "gold": "g0",
        },
        {
            "sample_index": 1,
            "condition": "ca",
            "status": "complete",
            "correct": True,
            "gold": "g1",
        },
        {
            "sample_index": 2,
            "condition": "ca",
            "status": "complete",
            "correct": False,
            "gold": "g2",
        },
        {
            "sample_index": 3,
            "condition": "ca",
            "status": "complete",
            "correct": True,
            "gold": "g3",
        },
    ]
    (run_a / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in recs_a))
    (run_a / "summary.json").write_text(
        json.dumps(
            {
                "sample_count": 4,
                "condition_statistics": {
                    "ca": {"completed": 4, "correct": 3, "accuracy": 0.75}
                },
            }
        )
    )

    recs_b = [
        {
            "sample_index": 0,
            "condition": "cb",
            "status": "complete",
            "correct": True,
            "gold": "g0",
        },
        {
            "sample_index": 1,
            "condition": "cb",
            "status": "complete",
            "correct": False,
            "gold": "g1",
        },
        {
            "sample_index": 2,
            "condition": "cb",
            "status": "complete",
            "correct": True,
            "gold": "g2",
        },
        {
            "sample_index": 3,
            "condition": "cb",
            "status": "complete",
            "correct": False,
            "gold": "g3",
        },
    ]
    (run_b / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in recs_b))
    (run_b / "summary.json").write_text(
        json.dumps(
            {
                "sample_count": 4,
                "condition_statistics": {
                    "cb": {"completed": 4, "correct": 2, "accuracy": 0.5}
                },
            }
        )
    )

    out_a = ConditionOutcome.from_run(run_a)
    out_b = ConditionOutcome.from_run(run_b)
    res = mcnemar(out_a, out_b)
    assert res.n11 == 1 and res.n10 == 2 and res.n01 == 1 and res.n00 == 0
    assert res.discordant_count == 3
    assert res.accuracy_diff == pytest.approx(0.25)
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_duplicate_sample_ids_raises() -> None:
    oa = ConditionOutcome(
        condition="ca", sample_ids=[0, 0, 1], correct=[True, False, True]
    )
    ob = ConditionOutcome(
        condition="cb", sample_ids=[0, 0, 1], correct=[True, True, True]
    )
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        mcnemar(oa, ob)
