"""Paired percentile estimates for accuracy and receiver cost."""

import numpy as np

COST_METRICS = (
    "receiver_latency_sec",
    "receiver_attention_pairs",
    "receiver_processed_positions",
)


def estimate(values, indices):
    if values is None:
        return {"mean": None, "ci_low": None, "ci_high": None}, None
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }, draws.tolist()


def cell_cost(rows, indices, valid):
    summary, distributions, arrays = {}, {}, {}
    for name in COST_METRICS:
        values = [r.get(name) for r in rows]
        # Saved runs without measurements cannot supply a measured cost curve.
        array = (
            np.array(values, dtype=float)
            if valid and all(v is not None for v in values)
            else None
        )
        stats, draws = estimate(array, indices)
        arrays[name], distributions[name] = array, draws
        summary.update(
            {
                f"mean_{name}": stats["mean"],
                f"{name}_ci_low": stats["ci_low"],
                f"{name}_ci_high": stats["ci_high"],
            }
        )
    return summary, distributions, arrays


def cost_difference(left, right, indices):
    result, draws = {}, {}
    for name in COST_METRICS:
        a, b = left[name], right[name]
        stats, distribution = estimate(
            a - b if a is not None and b is not None else None, indices
        )
        result[name], draws[name] = stats, distribution
    return result, draws


def threshold_rows(curves, config):
    rows = []
    for target in config["target_accuracies"]:
        for u in config["upstream_steps"]:
            for m in ("matched", "mismatched", "no_handoff"):
                candidates = [
                    r
                    for r in curves
                    if r["upstream_steps"] == u
                    and r["handoff_condition"] == m
                    and r["accuracy"] is not None
                    and r["accuracy"] >= target
                    and r["receiver_budget"] != "free"
                ]
                first = (
                    min(candidates, key=lambda r: r["receiver_budget"])
                    if candidates
                    else None
                )
                row = {
                    "upstream_steps": u,
                    "handoff_condition": m,
                    "target_accuracy": target,
                    "minimum_receiver_budget": first["receiver_budget"]
                    if first
                    else None,
                    "mean_generated_tokens_at_minimum_budget": first[
                        "mean_generated_tokens"
                    ]
                    if first
                    else None,
                }
                for name in COST_METRICS:
                    key = f"mean_{name}"
                    measured = [r for r in candidates if r[key] is not None]
                    best = (
                        min(measured, key=lambda r: (r[key], r["receiver_budget"]))
                        if measured
                        else None
                    )
                    row[f"{key}_at_minimum_budget"] = first[key] if first else None
                    row[f"minimum_{key}"] = best[key] if best else None
                    row[f"receiver_budget_at_minimum_{name}"] = (
                        best["receiver_budget"] if best else None
                    )
                rows.append(row)
    return rows
