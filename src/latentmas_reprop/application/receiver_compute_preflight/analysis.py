"""Strict paired bootstrap curves and descriptive artifact exports."""

import csv
import json
from collections import defaultdict

import numpy as np

from ..common.reporter import write_json

CONDITIONS = ("matched", "mismatched", "no_handoff")


def analyze(records, config):
    groups = defaultdict(dict)
    sample_ids = sorted({r["sample_id"] for r in records})
    expected_ids = config.get("sample_ids", sample_ids)
    if set(sample_ids) != set(expected_ids) or not sample_ids:
        raise ValueError("Sample set mismatch; paired analysis refused")
    for r in records:
        key = (r["upstream_steps"], r["handoff_condition"], r["receiver_budget"])
        if r["sample_id"] in groups[key]:
            raise ValueError("Duplicate sample cell")
        groups[key][r["sample_id"]] = r
    keys = [
        (u, m, b)
        for u in config["upstream_steps"]
        for m in CONDITIONS
        for b in config["receiver_budgets"]
    ]
    if set(groups) != set(keys) or any(
        set(g) != set(sample_ids) for g in groups.values()
    ):
        raise ValueError("Condition sample sets mismatch; paired analysis refused")
    count = config["bootstrap_count"]
    rng = np.random.default_rng(config["seed"])
    indices = rng.integers(0, len(sample_ids), size=(count, len(sample_ids)))
    distributions, curves, arrays = {}, [], {}
    for key in keys:
        rows = [groups[key][sid] for sid in sample_ids]
        valid = all(r["valid"] for r in rows)
        values = np.array([r["correct"] for r in rows], dtype=float)
        arrays[key] = values if valid else None
        draws = values[indices].mean(axis=1) if valid else None
        label = ":".join(map(str, key))
        distributions[label] = draws.tolist() if valid else None
        low, high = np.quantile(draws, [0.025, 0.975]) if valid else (None, None)
        curves.append(
            {
                "upstream_steps": key[0],
                "handoff_condition": key[1],
                "receiver_budget": key[2],
                "accuracy": float(values.mean()) if valid else None,
                "ci_low": float(low) if valid else None,
                "ci_high": float(high) if valid else None,
                "sample_count": len(rows),
                "invalid_count": sum(not r["valid"] for r in rows),
            }
        )
    comparisons = []
    for u in config["upstream_steps"]:
        for b in config["receiver_budgets"]:
            for name, left, right in (
                ("pairing_gain", "matched", "mismatched"),
                ("matched_vs_no_handoff", "matched", "no_handoff"),
                ("mismatched_vs_no_handoff", "mismatched", "no_handoff"),
            ):
                a, c = arrays[u, left, b], arrays[u, right, b]
                valid = a is not None and c is not None
                diff = a - c if valid else None
                draws = diff[indices].mean(axis=1) if valid else None
                label = f"{u}:{name}:{b}"
                distributions[label] = draws.tolist() if valid else None
                low, high = (
                    np.quantile(draws, [0.025, 0.975]) if valid else (None, None)
                )
                comparisons.append(
                    {
                        "upstream_steps": u,
                        "receiver_budget": b,
                        "comparison": name,
                        "difference": float(diff.mean()) if valid else None,
                        "ci_low": float(low) if valid else None,
                        "ci_high": float(high) if valid else None,
                    }
                )
    thresholds = []
    for target in config["target_accuracies"]:
        for u in config["upstream_steps"]:
            for m in CONDITIONS:
                candidates = [
                    r["receiver_budget"]
                    for r in curves
                    if r["upstream_steps"] == u
                    and r["handoff_condition"] == m
                    and r["accuracy"] is not None
                    and r["accuracy"] >= target
                    and r["receiver_budget"] != "free"
                ]
                thresholds.append(
                    {
                        "upstream_steps": u,
                        "handoff_condition": m,
                        "target_accuracy": target,
                        "minimum_receiver_budget": min(candidates)
                        if candidates
                        else None,
                    }
                )
    return {
        "curves": curves,
        "comparisons": comparisons,
        "thresholds": thresholds,
        "bootstrap_count": count,
        "seed": config["seed"],
        "sample_count": len(sample_ids),
    }, {
        "sample_ids": sample_ids,
        "resample_indices": indices.tolist(),
        "distributions": distributions,
        "ci_method": "paired percentile 95%",
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v
                for k, v in r.items()
            }
            for r in rows
        )


def export(directory, records, config):
    directory.mkdir(parents=True, exist_ok=True)
    metrics, bootstrap = analyze(records, config)
    write_json(directory / "metrics.json", metrics)
    write_json(directory / "bootstrap_statistics.json", bootstrap)
    write_csv(directory / "sample_matrix.csv", records)
    write_csv(directory / "budget_curves.csv", metrics["curves"])
    lines = [
        "# Receiver compute preflight",
        "",
        f"Samples: {metrics['sample_count']}",
        f"Bootstrap resamples: {metrics['bootstrap_count']}; paired percentile 95% CI.",
        "",
    ]
    for name in ("curves", "comparisons", "thresholds"):
        rows = metrics[name]
        fields = list(rows[0])
        lines.extend(
            [
                f"## {name}",
                "",
                "| " + " | ".join(fields) + " |",
                "| " + " | ".join(["---"] * len(fields)) + " |",
            ]
        )
        lines.extend(
            "| "
            + " | ".join("N/A" if r[k] is None else str(r[k]) for k in fields)
            + " |"
            for r in rows
        )
        lines.append("")
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics
