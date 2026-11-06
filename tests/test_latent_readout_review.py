"""Source audits and human labels remain distinct from evaluator correctness."""

import json

import pytest

from latentmas_reprop.application.latent_evaluation_audit import (
    audit_evaluations,
    explicit_choice,
    medqa_gold_audit,
)
from latentmas_reprop.application.latent_readout_review import (
    apply_annotations,
    review_template,
)


def test_medqa_mapping_checks_exact_answer_ambiguity_and_source_identity(tmp_path):
    dataset = tmp_path / "medqa.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "query": "q",
                    "answer": "Cancer",
                    "options": ["A. Other", "B. Different", "C. Cancer", "D. None"],
                }
            ]
        )
    )
    sample = tmp_path / "sample.json"
    sample.write_text(json.dumps({"index": 0, "question": "q", "gold": "c"}))
    assert medqa_gold_audit([sample], dataset)[0]["status"] == "verified"
    data = json.loads(dataset.read_text())
    data[0]["options"][0] = "A. Cancer risk"
    dataset.write_text(json.dumps(data))
    assert "ambiguous" in medqa_gold_audit([sample], dataset)[0]["reason"]
    data[0]["query"] = "another question"
    dataset.write_text(json.dumps(data))
    assert "question differs" in medqa_gold_audit([sample], dataset)[0]["reason"]
    assert (
        medqa_gold_audit([sample], tmp_path / "missing.json")[0]["status"]
        == "unavailable"
    )


@pytest.mark.parametrize(
    "text,choice",
    [
        (r"reasoning mentions A, B, C. Final: \boxed{D}", "d"),
        ("Reasoning A.\nThe final answer is C.", "c"),
        ("Consider A and B. Last test value was 12.", None),
        ("Answer: A\nStill considering C", None),
        (r"\boxed{Cancer}", None),
    ],
)
def test_explicit_choice_is_conservative(text, choice):
    assert explicit_choice(text) == choice


def test_evaluator_replay_flags_choice_evidence_without_replacing_score():
    rows = [
        {
            "status": "complete",
            "source_run_id": "run",
            "sample_id": 0,
            "condition": "a",
            "step": 0,
        }
    ]
    # Existing numeric fallback extracts 12; diagnostic sees the explicit C.
    record = {
        "evaluation_text": "Measurement was 12.\nAnswer: C",
        "prediction": "12",
        "correct": False,
        "gold": "c",
    }
    result = audit_evaluations(rows, {(0, "a", 0): record}, "medqa")[0]
    assert result["status"] == "verified"
    assert result["explicit_choice"] == "c"
    assert result["choice_disagrees"] and result["extraction_failed"]
    assert record["correct"] is False
    record["correct"] = True
    assert (
        audit_evaluations(rows, {(0, "a", 0): record}, "medqa")[0]["status"]
        == "mismatch"
    )


def row():
    return {
        "source_run_id": "run",
        "sample_id": 145,
        "condition": "a",
        "step": 1,
        "status": "complete",
        "gold": "c",
        "evaluator_correct": False,
        "semantic_correct": None,
        "review_status": "unreviewed",
        "question": "q",
        "output_path": "record.json",
        "prediction": None,
    }


def annotation(**updates):
    return {
        "source_run_id": "run",
        "sample_id": 145,
        "condition": "a",
        "step": 1,
        "semantic_correct": True,
        "reasoning_validity": "invalid",
        **updates,
    }


def test_annotations_use_full_identity_and_do_not_replace_evaluator(tmp_path):
    path = tmp_path / "annotations.json"
    entries = [annotation(), annotation(source_run_id="other")]
    path.write_text(json.dumps(entries))
    rows = [row()]
    result = apply_annotations(rows, path)
    assert [r["status"] for r in result] == ["applied", "unavailable"]
    assert rows[0]["semantic_correct"] is True
    assert rows[0]["reasoning_validity"] == "invalid"
    assert rows[0]["evaluator_correct"] is False
    assert json.loads(path.read_text()) == entries


def test_duplicate_invalid_blank_and_partial_annotations(tmp_path):
    path = tmp_path / "annotations.json"
    for entries, status in [
        ([annotation(), annotation()], "invalid"),
        ([annotation(semantic_correct="true")], "invalid"),
        ([annotation(reasoning_validity="maybe")], "invalid"),
        ([annotation(gold="d")], "invalid"),
        ([annotation(semantic_correct=None, reasoning_validity=None)], "unreviewed"),
    ]:
        path.write_text(json.dumps(entries))
        rows = [row()]
        assert all(r["status"] == status for r in apply_annotations(rows, path))
        assert rows[0]["semantic_correct"] is None
    assert apply_annotations([row()], None) == []
    path.write_text("{broken")
    assert apply_annotations([row()], path)[0]["status"] == "invalid"


def test_template_deduplicates_transition_endpoints():
    rows = [{**row(), "step": step} for step in range(3)]
    cases = [{"sample_id": 145, "condition": "a", "step": step} for step in (1, 2)]
    template = review_template(rows, cases)
    assert [r["step"] for r in template] == [0, 1, 2]
    assert all(r["semantic_correct"] is None for r in template)
