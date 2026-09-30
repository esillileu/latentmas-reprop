"""Read-only diagnostics from saved receiver observations."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

DIGITS = range(10)
CONDITIONS = ("drop", "latent_only/own", "latent_only/cross")


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count predictions against the actual source digit (target for drop)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drops = {}
    for record in records:
        if record["error"] is not None:
            continue
        condition = (
            "drop"
            if record["condition"] == "drop"
            else f"{record['context_mode']}/{record['condition']}"
        )
        if condition in CONDITIONS:
            groups[condition].append(record)
        if condition == "drop":
            drops[record["sample_index"]] = record

    result = {}
    for condition in CONDITIONS:
        group = groups[condition]
        if not group:
            continue
        matrix = [[0 for _ in DIGITS] for _ in DIGITS]
        probabilities = []
        deltas = []
        for record in group:
            true_digit = (
                record["target_digit"]
                if condition == "drop"
                else record["source_digit"]
            )
            predicted = record["predicted_digit"]
            if true_digit not in DIGITS or predicted not in DIGITS:
                raise ValueError(
                    f"Invalid digit in {condition}: {true_digit}, {predicted}"
                )
            matrix[true_digit][predicted] += 1
            probability = record["candidate_probabilities"][str(true_digit)]
            probabilities.append(probability)
            if condition != "drop" and record["sample_index"] in drops:
                baseline = drops[record["sample_index"]]["candidate_probabilities"][
                    str(true_digit)
                ]
                deltas.append(probability - baseline)
        counts = Counter(record["predicted_digit"] for record in group)
        top_digit, top_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        per_digit = [
            {
                "correct_count": matrix[digit][digit],
                "sample_count": sum(matrix[digit]),
                "accuracy": matrix[digit][digit] / sum(matrix[digit])
                if sum(matrix[digit])
                else None,
            }
            for digit in DIGITS
        ]
        result[condition] = {
            "sample_count": len(group),
            "accuracy": sum(matrix[d][d] for d in DIGITS) / len(group),
            "correct_class_probability": mean(probabilities),
            "probability_delta_vs_drop": mean(deltas) if deltas else None,
            "prediction_distribution": {
                str(digit): {
                    "count": counts[digit],
                    "fraction": counts[digit] / len(group),
                }
                for digit in DIGITS
            },
            "confusion_matrix": matrix,
            "per_digit_accuracy": per_digit,
            "unique_predictions": len(counts),
            "top_prediction": top_digit,
            "top_fraction": top_count / len(group),
        }
    drop_accuracy = result.get("drop", {}).get("accuracy")
    for values in result.values():
        values["accuracy_delta_vs_drop"] = (
            values["accuracy"] - drop_accuracy if drop_accuracy is not None else None
        )
    return result


def render_markdown(runs: list[dict[str, Any]]) -> str:
    def fmt(value: float | None) -> str:
        return f"{value:.4f}" if value is not None else "—"

    lines = [
        "# Receiver acquisition diagnostics",
        "",
        "Saved receiver sample results only; no inference was run. Accuracy, correct-class "
        "probability, and source-to-prediction counts are observations, not automatic "
        "acquisition labels.",
        "",
        "## Encoding → Acquisition results",
        "",
        "| Model | Steps | Drop Acc | Own Acc | Cross Acc | Own Δ Acc | Cross Δ Acc | "
        "Drop P(correct) | Own P(correct) | Cross P(correct) | Own ΔP(correct) | Cross ΔP(correct) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        cells = run["conditions"]
        if not all(c in cells for c in CONDITIONS):
            continue
        drop, own, cross = (cells[c] for c in CONDITIONS)
        lines.append(
            f"| {run['model']} | {run['latent_steps']} | {fmt(drop['accuracy'])} | "
            f"{fmt(own['accuracy'])} | {fmt(cross['accuracy'])} | "
            f"{fmt(own['accuracy_delta_vs_drop'])} | {fmt(cross['accuracy_delta_vs_drop'])} | "
            f"{fmt(drop['correct_class_probability'])} | "
            f"{fmt(own['correct_class_probability'])} | "
            f"{fmt(cross['correct_class_probability'])} | "
            f"{fmt(own['probability_delta_vs_drop'])} | "
            f"{fmt(cross['probability_delta_vs_drop'])} |"
        )
    lines += [
        "",
        "## Receiver diagnostic summary",
        "",
        "| Model | Steps | Condition | Accuracy | Δ Acc | ΔP(correct) | Unique Preds | Top Prediction | Top Fraction |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for condition in CONDITIONS:
            if condition not in run["conditions"]:
                continue
            values = run["conditions"][condition]
            lines.append(
                f"| {run['model']} | {run['latent_steps']} | {condition} | "
                f"{fmt(values['accuracy'])} | {fmt(values['accuracy_delta_vs_drop'])} | "
                f"{fmt(values['probability_delta_vs_drop'])} | "
                f"{values['unique_predictions']} | {values['top_prediction']} | "
                f"{values['top_fraction']:.2f} |"
            )
    lines += ["", "## Prediction distributions and source-digit detail", ""]
    for run in runs:
        lines += [f"### {run['model']}, steps={run['latent_steps']}", ""]
        for condition in CONDITIONS:
            if condition not in run["conditions"]:
                continue
            values = run["conditions"][condition]
            distribution = ", ".join(
                f"{digit}:{entry['fraction']:.0%}"
                for digit, entry in values["prediction_distribution"].items()
                if entry["count"]
            )
            lines.append(f"{condition}: {distribution}")
            lines += [
                "",
                f"<details><summary>{condition}: confusion matrix and per-digit accuracy</summary>",
                "",
                "Rows are true/source digits; columns are predicted digits.",
                "",
                "| True \\ Pred | "
                + " | ".join(map(str, DIGITS))
                + " | Correct / Total | Accuracy |",
                "|---:|" + "---:|" * 12,
            ]
            for digit, row in enumerate(values["confusion_matrix"]):
                stat = values["per_digit_accuracy"][digit]
                accuracy = (
                    "—" if stat["accuracy"] is None else f"{stat['accuracy']:.0%}"
                )
                lines.append(
                    f"| {digit} | "
                    + " | ".join(map(str, row))
                    + f" | {stat['correct_count']}/{stat['sample_count']} | {accuracy} |"
                )
            lines += ["", "</details>", ""]
    return "\n".join(lines) + "\n"
