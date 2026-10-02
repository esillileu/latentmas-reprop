"""Budget cutoffs cannot promote intermediate GSM8K numbers to final answers."""

import csv

import pytest

from latentmas_reprop.application.receiver_compute_preflight.answers import final_answer
from latentmas_reprop.application.receiver_compute_preflight.inference import (
    evaluate_prefix,
)
from latentmas_reprop.infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR


@pytest.mark.parametrize(
    "text",
    [
        "There are 42 apples so far",
        "42",
        r"The answer is \boxed{42",
        "Final answer: 42",
        "Final answer: 42.5",
        "Final answer:",
        r"\boxed{1} then \boxed{42",
        r"<think>Therefore \boxed{42}",
        r"<think>Final answer: 42.</think>Still computing 42",
    ],
)
def test_incomplete_final_answer_is_no_answer(text):
    assert final_answer(text) is None


@pytest.mark.parametrize(
    "text, answer",
    [
        (r"After computing 99, \boxed{42}", "42"),
        ("Final answer: 42.\n", "42"),
        ("The final answer is -42!", "-42"),
        ("Final answer: 1,234\n", "1234"),
        (r"<think>42 is intermediate</think>\boxed{43}", "43"),
    ],
)
def test_explicit_complete_final_answers(text, answer):
    assert final_answer(text) == answer


def test_cutoff_number_requires_delimiter_but_natural_endpoint_is_complete():
    assert final_answer("Final answer: 42", naturally_terminated=True) == "42"
    assert final_answer(r"\boxed{42}", thinking_open=True) is None
    assert final_answer(r"42</think>\boxed{42}", thinking_open=True) == "42"


def test_prefix_uses_ids_and_strict_scoring_at_each_cutoff():
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return "".join(ids)

    tokens = ["Work: 42. ", r"\boxed{", "4", "2", "}"]
    item = {"gold": "42"}
    for budget in (1, 2, 3, 4):
        row = evaluate_prefix(Tokenizer(), DEFAULT_EVALUATOR, item, tokens, budget)
        assert row["generated_token_ids"] == tokens[:budget]
        assert row["no_answer"] is True
        assert row["prediction"] is None
        assert row["correct"] is False
    complete = evaluate_prefix(Tokenizer(), DEFAULT_EVALUATOR, item, tokens, 5)
    assert complete["correct"] is True
    assert complete["no_answer"] is False
    row = evaluate_prefix(
        Tokenizer(), DEFAULT_EVALUATOR, item, ["Final answer: 42\n", "more"], 1
    )
    assert row["correct"] is True


def test_reanalysis_rescores_raw_records_and_exported_matrix(tmp_path):
    from latentmas_reprop.application.receiver_compute_preflight.analysis import export

    config = {
        "upstream_steps": [10],
        "receiver_budgets": [64, "free"],
        "target_accuracies": [0.5],
        "bootstrap_count": 10,
        "seed": 0,
        "sample_ids": ["a", "b"],
    }
    records = [
        {
            "sample_id": sid,
            "upstream_steps": 10,
            "handoff_condition": m,
            "receiver_budget": r,
            "raw_receiver_output": "Intermediate total: 42",
            "gold": "42",
            "correct": True,
            "prediction": "42",
            "valid": True,
            "generated_tokens": 3,
            "free_generated_tokens": 3,
            "free_cap_reached": False,
        }
        for sid in ["a", "b"]
        for m in ["matched", "mismatched", "no_handoff"]
        for r in [64, "free"]
    ]
    metrics = export(tmp_path, records, config)
    assert all(r["accuracy"] == 0 for r in metrics["curves"])
    assert all(r["mean_receiver_latency_sec"] is None for r in metrics["curves"])
    with (tmp_path / "sample_matrix.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert all(r["correct"] == "False" and r["no_answer"] == "True" for r in rows)
    assert all(r["correct"] is True for r in records)
