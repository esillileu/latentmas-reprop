"""Partial-result analysis of saved readouts, without loading models or KV."""

import json
from itertools import combinations

from ..domain.services.token_distance import token_levenshtein
from ..infrastructure.storage import load_json
from .common.statistics import mcnemar, paired_uncertainty
from .intervention_continuation.outcomes import ConditionOutcome
from .latent_readout_features import readout_features, readout_text
from .latent_trajectory_readout import is_readout_complete


def load_readouts(directory, metadata, tokenizer=None):
    """Snapshot paths first; represent missing, incomplete and invalid independently."""
    paths = set(directory.glob("samples/*/continuations/*/step_*.json"))
    samples = sorted((directory / "samples").glob("*/sample.json"))
    rows, records, tokens = [], {}, {}
    for sample_path in samples:
        sample = load_json(sample_path)
        markers = sample["thinking_token_ids"]
        for condition in "abcd":
            for step in range(
                metadata["latent_step_start"], metadata["latent_steps"] + 1
            ):
                path = (
                    sample_path.parent
                    / "continuations"
                    / condition
                    / f"step_{step:03d}.json"
                )
                row = {
                    "source_run_id": directory.name,
                    "benchmark": metadata["task"],
                    "split": metadata.get("split"),
                    "model": metadata["model_name"],
                    "alignment": metadata["latent_space_realign"],
                    "sample_id": sample["index"],
                    "question": sample.get("question"),
                    "condition": condition,
                    "trajectory_type": "without_think"
                    if condition in "ac"
                    else "with_think",
                    "step": step,
                    "status": "missing",
                    "error": None,
                    "output_path": str(path.resolve()),
                    "evaluator_correct": None,
                    "semantic_correct": None,
                    "review_status": "unreviewed",
                    "adjacent_token_distance": None,
                    "transition": None,
                }
                key = (sample["index"], condition, step)
                if path in paths:
                    try:
                        record = load_json(path)
                        if record["condition"] != condition or record["step"] != step:
                            raise ValueError("Continuation identity differs from path")
                        if not is_readout_complete(path):
                            row["status"] = "incomplete"
                        else:
                            row.update(
                                readout_features(record, markers), status="complete"
                            )
                            if str(record["gold"]) != str(sample["gold"]):
                                raise ValueError(
                                    "Continuation gold differs from sample"
                                )
                            records[key] = record
                    except (ValueError, KeyError, TypeError, OSError) as error:
                        row.update(
                            status="invalid", error=str(error), evaluator_correct=None
                        )
                rows.append(row)
                if row["status"] != "complete":
                    continue
                if tokenizer is not None:
                    tokens[key] = tokenizer.encode(
                        readout_text(records[key], tokenizer), add_special_tokens=False
                    )
                previous = (sample["index"], condition, step - 1)
                if previous in records:
                    before, after = (
                        records[previous]["correct"],
                        records[key]["correct"],
                    )
                    row["transition"] = f"{int(before)}→{int(after)}"
                    if previous in tokens and key in tokens:
                        left, right = tokens[previous], tokens[key]
                        row["adjacent_token_distance"] = token_levenshtein(
                            left, right
                        ) / max(len(left), len(right), 1)
    return rows, records


def summarize_readouts(rows):
    """Keep within-item step fractions separate from across-item step accuracy."""
    groups, cells = {}, {}
    for row in rows:
        groups.setdefault((row["sample_id"], row["condition"]), []).append(row)
        cells.setdefault((row["condition"], row["step"]), []).append(row)
    trajectories = []
    for (sample, condition), group in groups.items():
        available = [r for r in group if r["status"] == "complete"]
        transitions = [r["transition"] for r in available if r["transition"]]
        correct = sum(r["evaluator_correct"] for r in available)
        streak, longest = 0, 0
        for row in group:
            streak = (
                streak + 1
                if row["status"] == "complete" and row["evaluator_correct"]
                else 0
            )
            longest = max(longest, streak)

        gains, losses = transitions.count("0→1"), transitions.count("1→0")
        pattern = "unavailable"
        if available:
            pattern = (
                "unstable"
                if gains and losses
                else "improving"
                if gains
                else "worsening"
                if losses
                else "unchanged"
            )
        trajectories.append(
            {
                "sample_id": sample,
                "condition": condition,
                "completed": len(available),
                "expected": len(group),
                "correct_step_fraction": correct / len(available)
                if available
                else None,
                "first_correct_step": next(
                    (r["step"] for r in available if r["evaluator_correct"]), None
                ),
                "wrong_to_correct": gains,
                "correct_to_wrong": losses,
                "pattern": pattern,
                "longest_correct_streak": longest,
            }
        )
    accuracies = []
    for (condition, step), group in cells.items():
        available = [r for r in group if r["status"] == "complete"]
        accuracies.append(
            {
                "condition": condition,
                "step": step,
                "n_samples": len(available),
                "expected_samples": len(group),
                "sample_ids": json.dumps([r["sample_id"] for r in available]),
                "accuracy": sum(r["evaluator_correct"] for r in available)
                / len(available)
                if available
                else None,
            }
        )
    return trajectories, accuracies


def compare_conditions(rows):
    """Pair only complete observations of the same item at the same step."""
    results = []
    for step in sorted({r["step"] for r in rows}):
        cells = {
            c: {
                r["sample_id"]: r
                for r in rows
                if r["step"] == step
                and r["condition"] == c
                and r["status"] == "complete"
            }
            for c in "abcd"
        }
        for left, right in combinations("abcd", 2):
            ids = sorted(cells[left].keys() & cells[right].keys())
            result = {
                "step": step,
                "left": left,
                "right": right,
                "n_pairs": len(ids),
                "unpaired_sample_ids": json.dumps(
                    sorted(cells[left].keys() ^ cells[right].keys())
                ),
                "sample_ids": json.dumps(ids),
                "accuracy_diff": None,
                "n11": None,
                "n10": None,
                "n01": None,
                "n00": None,
            }
            if ids:
                outcomes = [
                    ConditionOutcome(
                        c,
                        ids,
                        [cells[c][i]["evaluator_correct"] for i in ids],
                        golds=[cells[c][i]["gold"] for i in ids],
                    )
                    for c in (left, right)
                ]
                stats = mcnemar(*outcomes)
                result.update(
                    stats.contingency_table,
                    **paired_uncertainty(stats),
                    accuracy_diff=stats.accuracy_diff,
                )
            results.append(result)
    return results
