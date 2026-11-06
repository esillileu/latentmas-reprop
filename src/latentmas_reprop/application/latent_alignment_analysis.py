"""Descriptive alignment comparisons with explicit source/protocol checks."""

import json

from .common.statistics import mcnemar
from .intervention_continuation.outcomes import ConditionOutcome

# Alignment is deliberately absent; all other inference settings must agree.
PROTOCOL_FIELDS = (
    "task",
    "split",
    "model_name",
    "model_revision",
    "tokenizer",
    "tokenizer_revision",
    "dtype",
    "chat_template",
    "prompt",
    "initial_input",
    "initial_role",
    "seed",
    "max_new_tokens",
    "effective_decoding",
)


def alignment_protocol_mismatches(left_metadata, right_metadata):
    mismatches = [
        key
        for key in PROTOCOL_FIELDS
        if left_metadata.get(key) != right_metadata.get(key)
    ]
    if left_metadata["latent_space_realign"] == right_metadata["latent_space_realign"]:
        mismatches.append("alignment must differ")
    left_identity = left_metadata.get("state_identity", {}).get("tokenizer")
    right_identity = right_metadata.get("state_identity", {}).get("tokenizer")
    if left_identity != right_identity:
        mismatches.append("tokenizer identity")
    return mismatches


def compare_alignment(left_rows, right_rows, left_metadata, right_metadata):
    mismatches = alignment_protocol_mismatches(left_metadata, right_metadata)
    cells = []
    for condition in "abcd":
        for step in sorted({r["step"] for r in left_rows + right_rows}):
            left = {
                r["sample_id"]: r
                for r in left_rows
                if r["condition"] == condition and r["step"] == step
            }
            right = {
                r["sample_id"]: r
                for r in right_rows
                if r["condition"] == condition and r["step"] == step
            }
            ids = sorted(
                i
                for i in left.keys() & right.keys()
                if left[i]["status"] == right[i]["status"] == "complete"
            )
            result = {
                "condition": condition,
                "step": step,
                "left_run": left_rows[0]["source_run_id"] if left_rows else None,
                "right_run": right_rows[0]["source_run_id"] if right_rows else None,
                "left_alignment": left_metadata["latent_space_realign"],
                "right_alignment": right_metadata["latent_space_realign"],
                "status": "incompatible"
                if mismatches
                else "complete"
                if ids
                else "pending",
                "reason": ", ".join(mismatches),
                "n_pairs": 0,
                "sample_ids": "[]",
                "unpaired_sample_ids": json.dumps(
                    sorted((set(left) | set(right)) - set(ids))
                ),
                "accuracy_diff": None,
                "n11": None,
                "n10": None,
                "n01": None,
                "n00": None,
            }
            if not mismatches and ids:
                if any(
                    left[i]["question"] != right[i]["question"]
                    or left[i]["gold"] != right[i]["gold"]
                    for i in ids
                ):
                    result.update(
                        status="incompatible", reason="question or gold differs"
                    )
                else:
                    outcomes = [
                        ConditionOutcome(
                            name, ids, [data[i]["evaluator_correct"] for i in ids]
                        )
                        for name, data in (("left", left), ("right", right))
                    ]
                    stats = mcnemar(*outcomes)
                    result.update(
                        stats.contingency_table,
                        n_pairs=len(ids),
                        accuracy_diff=stats.accuracy_diff,
                        sample_ids=json.dumps(ids),
                    )
            cells.append(result)
    return cells
