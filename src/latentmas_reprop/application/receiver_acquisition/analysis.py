"""Read-only diagnostics from saved receiver observations."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .inference import enrich_condition
from .statistics import average, entropy, mutual_information, probability_metrics

DIGITS = range(10)
CONDITIONS = ("drop", "latent_only/own", "latent_only/cross")


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def select_receiver_runs(
    candidates: list[dict[str, Any]], designated_ids: set[str]
) -> list[dict[str, Any]]:
    """Select one complete Receiver run per model and step, rejecting ambiguity."""
    available_ids = {run["run_id"] for run in candidates}
    unknown_ids = designated_ids - available_ids
    if unknown_ids:
        raise ValueError(
            f"Designated Receiver run IDs not found: {sorted(unknown_ids)}"
        )
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in candidates:
        groups[(run["model"], run["latent_steps"])].append(run)
    selected = []
    for (model, steps), group in sorted(groups.items()):
        designated = [
            run for run in group if run["run_id"] in designated_ids or run["designated"]
        ]
        if len(designated) > 1:
            ids = sorted(run["run_id"] for run in designated)
            raise ValueError(
                f"Ambiguous designated Receiver runs for {model}, steps={steps}: {ids}"
            )
        if designated:
            if designated[0]["sample_count"] != 100 or any(
                designated[0]["condition_counts"].get(c) != 100 for c in CONDITIONS
            ):
                raise ValueError(
                    "Designated Receiver run must contain 100 samples in each condition"
                )
            selected.append(designated[0])
            continue
        canonical = [
            run
            for run in group
            if run["sample_count"] == 100
            and all(
                run["condition_counts"].get(condition) == 100
                for condition in CONDITIONS
            )
        ]
        if len(canonical) > 1:
            ids = sorted(run["run_id"] for run in canonical)
            raise ValueError(
                f"Ambiguous canonical Receiver runs for {model}, steps={steps}: {ids}"
            )
        if canonical:
            selected.append(canonical[0])
    return selected


def analyze_records(
    records: list[dict[str, Any]], *, permutations: int = 5000, seed: int = 0
) -> dict[str, Any]:
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
            probability = record.get("candidate_probabilities", {}).get(str(true_digit))
            if probability is not None:
                probabilities.append(probability)
            if condition != "drop" and record["sample_index"] in drops:
                baseline = (
                    drops[record["sample_index"]]
                    .get("candidate_probabilities", {})
                    .get(str(true_digit))
                )
                if probability is not None and baseline is not None:
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
            "correct_class_probability": mean(probabilities) if probabilities else None,
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
    rng = np.random.default_rng(seed)
    for condition, values in result.items():
        group = groups[condition]
        labels = np.array(
            [
                r["target_digit"] if condition == "drop" else r["source_digit"]
                for r in group
            ]
        )
        predictions = np.array([r["predicted_digit"] for r in group])
        source_counts = np.bincount(labels, minlength=10)
        values["source_digit_counts"] = source_counts.tolist()
        values["source_distribution"] = (source_counts / len(group)).tolist()
        values["balanced_accuracy"] = mean(
            s["accuracy"]
            for s in values["per_digit_accuracy"]
            if s["accuracy"] is not None
        )
        values["prediction_entropy"] = entropy(
            np.bincount(predictions, minlength=10) / len(group)
        )
        pairs = [
            (r, drops[r["sample_index"]]) for r in group if r["sample_index"] in drops
        ]
        values["paired_sample_count"] = len(pairs)
        values["argmax_changed_fraction"] = (
            mean(r["predicted_digit"] != d["predicted_digit"] for r, d in pairs)
            if pairs
            else None
        )
        values["matched_drop_accuracy"] = (
            mean(
                d["predicted_digit"]
                == (r["target_digit"] if condition == "drop" else r["source_digit"])
                for r, d in pairs
            )
            if pairs
            else None
        )
        values["accuracy_delta_vs_drop"] = (
            mean(
                (
                    r["predicted_digit"]
                    == (r["target_digit"] if condition == "drop" else r["source_digit"])
                )
                - (
                    d["predicted_digit"]
                    == (r["target_digit"] if condition == "drop" else r["source_digit"])
                )
                for r, d in pairs
            )
            if pairs
            else None
        )
        metrics = defaultdict(list)
        paired_metrics = defaultdict(list)
        js = []
        for r in group:
            label = r["target_digit"] if condition == "drop" else r["source_digit"]
            current = probability_metrics(r, label)
            if current is None:
                continue
            for key, value in current.items():
                if value is not None:
                    metrics[key].append(value)
            baseline = (
                probability_metrics(drops[r["sample_index"]], label)
                if r["sample_index"] in drops
                else None
            )
            if baseline is not None:
                for key in current:
                    if current[key] is not None and baseline[key] is not None:
                        paired_metrics[key].append(
                            np.asarray(current[key]) - np.asarray(baseline[key])
                        )
                p, q = (
                    np.array(current["normalized_digit_probabilities"]),
                    np.array(baseline["normalized_digit_probabilities"]),
                )
                js.append(entropy((p + q) / 2) - (entropy(p) + entropy(q)) / 2)
        for key in (
            "digit_probability_mass",
            "normalized_digit_probabilities",
            "source_label_nll",
            "brier_score",
        ):
            values[key] = average(metrics[key])
            values[key + "_delta_vs_drop"] = average(paired_metrics[key])
        values["probability_sample_count"] = len(metrics["digit_probability_mass"])
        values["probability_paired_sample_count"] = len(js)
        values["js_divergence_vs_drop"] = mean(js) if js else None
        if condition != "drop":
            observed = mutual_information(labels, predictions)
            null = [
                mutual_information(rng.permutation(labels), predictions)
                for _ in range(permutations)
            ]
            source_entropy = entropy(source_counts / len(group))
            values["dependency"] = {
                "mutual_information": observed,
                "source_entropy": source_entropy,
                "normalized_mi": observed / source_entropy if source_entropy else None,
                "permutation_p_value": (1 + sum(v >= observed for v in null))
                / (1 + len(null))
                if null
                else None,
                "permutation_statistics": null,
                "seed": seed,
                "permutations": permutations,
            }
        enrich_condition(
            values,
            group,
            drop=condition == "drop",
            permutations=permutations,
            seed=seed,
        )
    return result
