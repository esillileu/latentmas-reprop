"""Explicit, complete GSM8K final answers for cutoff evaluation."""

import re
from itertools import pairwise

from .cost import prefix_cost

NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
BOX = re.compile(r"\\boxed\{\s*(" + NUMBER + r")\s*\}")
MARKER = re.compile(
    r"\\boxed\{|(?:the\s+)?final\s+answer\s*(?::|=|is\b)", re.IGNORECASE
)
FINAL = re.compile(
    r"(?:the\s+)?final\s+answer\s*(?::|=|is\b)\s*\$?(" + NUMBER + r")"
    r"(?P<end>\$|[.!](?!\d)|\s*\n|\s*$)",
    re.IGNORECASE,
)


def final_answer(text, naturally_terminated=False, thinking_open=False):
    """Require a closed box or delimited final-answer number outside thinking."""
    if thinking_open or "<think>" in text or "</think>" in text:
        if "</think>" not in text:
            return None
        text = text.rsplit("</think>", 1)[1]
    matches = [(m.start(), m.group(1)) for m in BOX.finditer(text)]
    for match in FINAL.finditer(text):
        # An undelimited number at a budget cutoff might still gain digits.
        if (
            match.group("end").strip()
            or "\n" in match.group("end")
            or naturally_terminated
        ):
            matches.append((match.start(), match.group(1)))
    if not matches:
        return None
    last = max(matches)
    if any(m.start() > last[0] for m in MARKER.finditer(text)):
        return None
    return last[1].replace(",", "")


def evaluate_final_answer(
    evaluator, text, gold, naturally_terminated=False, thinking_open=False
):
    answer = final_answer(text, naturally_terminated, thinking_open)
    if answer is None:
        return {
            "prediction": None,
            "correct": False,
            "error_msg": "no_answer",
            "no_answer": True,
            "final_answer_complete": False,
        }
    prediction, correct, error = evaluator.evaluate(
        "gsm8k", rf"\boxed{{{answer}}}", gold
    )
    return {
        "prediction": prediction,
        "correct": bool(correct),
        "error_msg": error,
        "no_answer": False,
        "final_answer_complete": True,
    }


def trajectory_diagnostics(row):
    attempts = row["free_attempts"]
    for first, second in pairwise(attempts):
        if (
            second["generated_token_ids"][: len(first["generated_token_ids"])]
            != first["generated_token_ids"]
        ):
            raise ValueError(
                "Expanded free trajectory differs from initial greedy prefix"
            )
    final_ids = attempts[-1]["generated_token_ids"]
    checks = row["prefix_verification_attempts"]
    for check in checks:
        if check["generated_token_ids"] != final_ids[: check["cap"]] or any(
            check[key] != row[key]
            for key in (
                "receiver_prompt",
                "receiver_input_ids",
                "receiver_thinking_open",
            )
        ):
            raise ValueError(
                f"Greedy prefix differs from capped generation at R={check['cap']}"
            )
    return {
        "prefix_verified": bool(checks),
        "free_retry_count": len(attempts) - 1,
        "receiver_retry_latency_sec": sum(
            a["receiver_latency_sec"] for a in attempts[:-1]
        ),
    }


def reevaluate_records(records, evaluator):
    """Apply current final-answer policy to saved text without re-tokenization."""
    evaluated = []
    for r in records:
        natural = r.get(
            "free_naturally_terminated", not r.get("free_cap_reached", True)
        )
        prompt = r.get("receiver_prompt", "")
        thinking_open = r.get(
            "receiver_thinking_open", prompt.rfind("<think>") > prompt.rfind("</think>")
        )
        row = r | evaluate_final_answer(
            evaluator,
            r["raw_receiver_output"],
            r["gold"],
            natural and r["generated_tokens"] == r["free_generated_tokens"],
            thinking_open,
        )
        if r["receiver_budget"] == "free" and not natural:
            row.update(valid=False, pathological=True)
        if all(
            key in r
            for key in (
                "receiver_cache_positions",
                "receiver_prompt_tokens",
                "receiver_budget_latency_sec",
                "receiver_latency_sec",
            )
        ):
            row.update(prefix_cost(r, row["generated_tokens"], row["receiver_budget"]))
            row["receiver_generated_tokens_per_sec"] = (
                row["generated_tokens"] / row["receiver_latency_sec"]
                if row["receiver_latency_sec"]
                else 0.0
            )
        if "donor_id" in r:
            delta = (
                r["donor_sequence_length"] - r["recipient_sequence_length"]
                if r["donor_id"]
                else None
            )
            row.update(
                donor_length_delta=delta,
                donor_length_abs_delta=abs(delta) if delta is not None else None,
            )
        if "free_attempts" in r:
            row.update(trajectory_diagnostics(r))
        evaluated.append(row)
    return evaluated
