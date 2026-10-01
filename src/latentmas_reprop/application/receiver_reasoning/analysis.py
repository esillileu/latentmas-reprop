"""Descriptive cells and sample-paired percentile bootstrap statistics."""

import json
from collections import Counter
from statistics import mean, median

import numpy as np

from .integrity import inspect_records


def prediction_key(value):
    return (
        "<PARSE_FAILURE>" if value is None or str(value).strip() == "" else str(value)
    )


def reached_token_limit(row):
    tokens = row.get("receiver_generated_tokens")
    limit = row.get("config", {}).get(
        "answer_only_max_new_tokens"
        if row.get("receiver_mode") == "answer_only"
        else "max_new_tokens"
    )
    return bool(row.get("receiver_token_limit_reached")) or (
        isinstance(tokens, int) and isinstance(limit, int) and tokens >= limit
    )


def cell_statistics(rows):
    predictions = Counter(prediction_key(r.get("prediction")) for r in rows)
    tokens = [
        r["receiver_generated_tokens"]
        for r in rows
        if isinstance(r.get("receiver_generated_tokens"), int)
        and r["receiver_generated_tokens"] >= 0
    ]
    n = len(rows)
    correct = sum(r.get("correct") is True for r in rows)
    token_summary = {
        "mean": mean(tokens) if tokens else None,
        "median": median(tokens) if tokens else None,
        "min": min(tokens) if tokens else None,
        "max": max(tokens) if tokens else None,
    }
    frequencies = [
        {"prediction": p, "count": c, "ratio": c / n}
        for p, c in predictions.most_common()
    ]
    return {
        "sample_count": n,
        "correct_count": correct,
        "accuracy": correct / n if n else None,
        "generated_tokens": token_summary,
        "parse_failure_count": sum(
            r.get("prediction") is None or str(r.get("prediction", "")).strip() == ""
            for r in rows
        ),
        "empty_output_count": sum(
            not str(r.get("raw_receiver_output") or "").strip() for r in rows
        ),
        "truncation_count": sum(reached_token_limit(r) for r in rows),
        "unique_prediction_count": sum(p != "<PARSE_FAILURE>" for p in predictions),
        "top_prediction": frequencies[0] if frequencies else None,
        "top_10_predictions": frequencies[:10],
    }, frequencies


def analyze_records(records, config, metadata, *, bootstrap_count=5000, seed=0):
    if bootstrap_count < 1:
        raise ValueError("bootstrap_count must be positive")
    cells, indexed, sets, problems, integrity = inspect_records(
        records, config, metadata.get("params", {})
    )
    stats, distribution = {}, []
    for key, rows in cells.items():
        stats[key], frequencies = cell_statistics(rows)
        distribution.extend({"cell": key, **row} for row in frequencies)
    metric_specs = {
        "D_low": {"low_free": 1, "low_answer_only": -1},
        "D_high": {"high_free": 1, "high_answer_only": -1},
        "Delta_answer_only": {"high_answer_only": 1, "low_answer_only": -1},
        "Delta_free": {"high_free": 1, "low_free": -1},
        "substitution_signal": {
            "low_free": 1,
            "low_answer_only": -1,
            "high_free": -1,
            "high_answer_only": 1,
        },
    }
    # Reuse the same draw indices for every eligible metric with the same sample set.
    rng = np.random.default_rng(seed)
    draws_by_ids = {}
    derived, bootstrap = {}, {}
    for name, weights in metric_specs.items():
        keys = list(weights)
        reasons = [reason for key in keys for reason in problems[key]]
        if len({frozenset(sets[key]) for key in keys}) != 1:
            reasons.append("Compared cells have different sample sets")
        ids = sorted(sets[keys[0]])
        if not ids:
            reasons.append("No paired samples")
        if reasons:
            derived[name] = {
                "estimate": None,
                "ci95": None,
                "paired_sample_count": 0,
                "unavailable_reasons": list(dict.fromkeys(reasons)),
                "paired_transitions": None,
            }
            bootstrap[name] = {"sample_ids": [], "replicates": []}
            continue
        key_ids = tuple(ids)
        if key_ids not in draws_by_ids:
            draws_by_ids[key_ids] = rng.integers(
                0, len(ids), size=(bootstrap_count, len(ids))
            )
        values = np.array(
            [
                sum(
                    weight * int(indexed[key][sid]["correct"])
                    for key, weight in weights.items()
                )
                for sid in ids
            ],
            dtype=float,
        )
        replicates = values[draws_by_ids[key_ids]].mean(axis=1)
        transitions = None
        if len(keys) == 2:
            positive = next(key for key, weight in weights.items() if weight == 1)
            negative = next(key for key, weight in weights.items() if weight == -1)
            counts = Counter(
                (indexed[positive][sid]["correct"], indexed[negative][sid]["correct"])
                for sid in ids
            )
            transitions = {
                "positive_cell": positive,
                "negative_cell": negative,
                "both_correct": counts[True, True],
                "positive_only_correct": counts[True, False],
                "negative_only_correct": counts[False, True],
                "both_wrong": counts[False, False],
            }
        derived[name] = {
            "estimate": float(values.mean()),
            "ci95": np.quantile(replicates, [0.025, 0.975]).tolist(),
            "paired_sample_count": len(ids),
            "paired_transitions": transitions,
            "unavailable_reasons": [],
        }
        bootstrap[name] = {"sample_ids": ids, "replicates": replicates.tolist()}
    matrix = []
    for sid in sorted(set().union(*sets.values())):
        existing = [rows[sid] for rows in indexed.values() if sid in rows]
        row = {
            "sample_id": sid,
            "sample_index": existing[0].get("sample_index") if existing else None,
            "gold": existing[0].get("gold") if existing else None,
            "question": existing[0].get("question") if existing else None,
        }
        for key in cells:
            record = indexed[key].get(sid)
            for field in (
                "prediction",
                "correct",
                "receiver_generated_tokens",
            ):
                row[f"{key}_{field}"] = record.get(field) if record else None
        matrix.append(row)
    matrix.sort(
        key=lambda row: (
            row["sample_index"] is None,
            row["sample_index"] or 0,
            row["sample_id"],
        )
    )
    subsets = {
        name: []
        for name in (
            "low_disagreement",
            "high_disagreement",
            "answer_only_wrong_to_correct",
            "answer_only_correct_to_wrong",
            "repeated_predictions",
        )
    }
    for row in matrix:
        sid = row["sample_id"]
        for level in ("low", "high"):
            values = [
                row[f"{level}_{mode}_correct"] for mode in ("answer_only", "free")
            ]
            if (
                derived[f"D_{level}"]["estimate"] is not None
                and all(isinstance(v, bool) for v in values)
                and values[0] != values[1]
            ):
                subsets[f"{level}_disagreement"].append(sid)
        low, high = row["low_answer_only_correct"], row["high_answer_only_correct"]
        if (
            derived["Delta_answer_only"]["estimate"] is not None
            and low is False
            and high is True
        ):
            subsets["answer_only_wrong_to_correct"].append(sid)
        if (
            derived["Delta_answer_only"]["estimate"] is not None
            and low is True
            and high is False
        ):
            subsets["answer_only_correct_to_wrong"].append(sid)
        repeated = [
            key
            for key in cells
            if row[f"{key}_prediction"] is not None
            and any(
                d["cell"] == key
                and d["count"] >= 2
                and d["prediction"] == prediction_key(row[f"{key}_prediction"])
                for d in distribution
            )
        ]
        if repeated:
            subsets["repeated_predictions"].append(
                {"sample_id": sid, "cells": repeated}
            )
    raw_examples = []
    for key, rows in cells.items():
        selected = [
            r
            for r in rows
            if any(
                r.get("sample_id") in subsets[f"{level}_disagreement"]
                for level in ("low", "high")
            )
        ]
        if not selected:
            selected = rows
        if selected:
            r = selected[0]
            raw_examples.append(
                {
                    "cell": key,
                    "sample_id": r["sample_id"],
                    "raw_output": r.get("raw_receiver_output", "")[:1200],
                    "excerpt_truncated": len(r.get("raw_receiver_output", "")) > 1200,
                }
            )
    metrics = {
        "run": metadata,
        "config": config,
        "integrity": integrity,
        "cells": stats,
        "derived": derived,
        "subsets": subsets,
        "selected_raw_outputs": raw_examples,
        "repeated_prediction_definition": "Same parsed prediction occurs at least twice within a cell; parse failures excluded.",
    }
    statistics = {
        "method": "sample-paired percentile bootstrap",
        "confidence_level": 0.95,
        "bootstrap_count": bootstrap_count,
        "seed": seed,
        "metrics": bootstrap,
    }
    # Detect non-JSON numeric values before writing artifacts.
    json.dumps(metrics, allow_nan=False)
    return metrics, matrix, distribution, statistics
