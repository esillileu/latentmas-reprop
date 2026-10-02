"""Aggregate recorded experimental costs, paired behavior and execution health."""

from collections import Counter, defaultdict
from statistics import mean, median

from .analysis import cell_statistics
from .diagnostics import (
    execution_counts,
    inference_costs,
    numeric_statistics,
)


def summarize_receiver_records(records: list[dict], steps: list[int]) -> dict:
    """Report descriptive aggregates without inferring causal effects."""
    levels = {}
    for step in (
        [0] if any(r["upstream_latent_steps"] == 0 for r in records) else []
    ) + steps:
        cells = {}
        paired = {}
        for mode in ("answer_only", "free"):
            rows = [
                r
                for r in records
                if r["upstream_latent_steps"] == step and r["receiver_mode"] == mode
            ]
            counts = Counter(r["prediction"] for r in rows)
            tokens = [r["receiver_generated_tokens"] for r in rows]
            cells[mode] = cell_statistics(rows)[0] | {
                "sample_count": len(rows),
                "accuracy": mean(r["correct"] for r in rows),
                "mean_generated_tokens": mean(tokens),
                "median_generated_tokens": median(tokens),
                "prediction_unique_count": len(counts),
                "prediction_frequencies": dict(counts.most_common()),
                "token_limit_count": sum(
                    r["receiver_token_limit_reached"] for r in rows
                ),
            }
            for row in rows:
                paired.setdefault(row["sample_id"], {})[mode] = row["correct"]
        transitions = Counter(
            "both_correct"
            if p["free"] and p["answer_only"]
            else "free_only_correct"
            if p["free"]
            else "answer_only_correct"
            if p["answer_only"]
            else "both_incorrect"
            for p in paired.values()
        )
        levels[str(step)] = {
            "cells": cells,
            "paired_correctness": {
                key: transitions[key]
                for key in (
                    "both_correct",
                    "free_only_correct",
                    "answer_only_correct",
                    "both_incorrect",
                )
            },
            "D": cells["free"]["accuracy"] - cells["answer_only"]["accuracy"],
        }
    summary = {"upstream_levels": levels, **paired_diagnostics(records, steps)}
    if len(steps) == 2:
        summary["substitution_signal"] = (
            levels[str(steps[0])]["D"] - levels[str(steps[1])]["D"]
        )
    return summary


def summarize_execution(records, n_samples, eval_seconds, model, run_memory, steps):
    contexts = {r["context_id"]: r for r in records}
    runtime = {
        "model_load_time_sec": float(getattr(model, "load_time_sec", 0.0)),
        "eval_time_sec": eval_seconds,
        "total_time_sec": eval_seconds,
        "time_per_sample_sec": eval_seconds / n_samples if n_samples else 0.0,
        **inference_costs(records),
        **run_memory,
    }
    runtime["output_tokens_mean"] = (
        runtime["output_tokens_total"] / len(records) if records else 0.0
    )
    runtime["latent_steps_mean"] = (
        runtime["latent_steps_total"] / n_samples if n_samples else 0.0
    )
    expected_cells = 2 * (1 + len(steps))
    execution = {
        "n_samples": n_samples,
        **execution_counts(records, n_samples * expected_cells),
    }
    context_stats = {}
    for step in sorted({r["upstream_latent_steps"] for r in contexts.values()}):
        group = [r for r in contexts.values() if r["upstream_latent_steps"] == step]
        context_stats[str(step)] = {
            field: numeric_statistics([r[field] for r in group])
            for field in (
                "upstream_build_latency_sec",
                "upstream_source_sequence_length",
                "upstream_source_cache_bytes",
                "handoff_positions",
                "cache_bytes",
                "cache_num_layers",
                "upstream_latent_steps_executed",
                "upstream_prompt_tokens",
                "upstream_agent_count",
                "upstream_peak_vram_bytes",
                "upstream_peak_vram_reserved_bytes",
                "upstream_incremental_peak_vram_bytes",
                "upstream_incremental_peak_vram_reserved_bytes",
            )
        }
        context_stats[str(step)]["context_count"] = len(group)
    return {
        "runtime": runtime,
        "execution": execution,
        "upstream_statistics": context_stats,
    }


def paired_diagnostics(records, steps):
    by_cell = defaultdict(dict)
    by_sample = defaultdict(dict)
    for row in records:
        key = f"{row['upstream_latent_steps']}_{row['receiver_mode']}"
        by_cell[key][row["sample_id"]] = row
        by_sample[row["sample_id"]][key] = row["correct"]
    comparisons = {}
    pairs = [(f"{s}_free", f"{s}_answer_only") for s in by_step(records)]
    if len(steps) == 2:
        pairs += [
            (f"{steps[1]}_{m}", f"{steps[0]}_{m}") for m in ("answer_only", "free")
        ]
    if 0 in by_step(records):
        pairs += [
            (f"{s}_{m}", f"0_{m}") for s in steps for m in ("answer_only", "free")
        ]
    for positive, negative in pairs:
        a, b = by_cell[positive], by_cell[negative]
        if not a or set(a) != set(b):
            continue
        outcomes = Counter((a[s]["correct"], b[s]["correct"]) for s in a)
        comparisons[f"{positive}_vs_{negative}"] = {
            "paired_sample_count": len(a),
            "accuracy_delta": mean(
                int(a[s]["correct"]) - int(b[s]["correct"]) for s in a
            ),
            "answer_change_rate": mean(
                a[s]["prediction"] != b[s]["prediction"] for s in a
            ),
            "both_correct": outcomes[True, True],
            "both_incorrect": outcomes[False, False],
            "rescue_count": outcomes[True, False],
            "harm_count": outcomes[False, True],
        }
    patterns = Counter(
        " ".join(f"{k}={int(v)}" for k, v in sorted(cells.items()))
        for cells in by_sample.values()
    )
    return {"paired_comparisons": comparisons, "correctness_patterns": dict(patterns)}


def by_step(records):
    return sorted({r["upstream_latent_steps"] for r in records})


def log_summary_metrics(summary):
    """Publish comparison metrics; keep detailed distributions in artifacts."""
    metrics = {}
    if "substitution_signal" in summary:
        metrics["substitution_signal"] = summary["substitution_signal"]
    for step, level in summary.get("upstream_levels", {}).items():
        metrics[f"steps_{step}_D"] = level["D"]
        for mode, cell in level["cells"].items():
            for name in (
                "accuracy",
                "mean_generated_tokens",
                "median_generated_tokens",
                "prediction_unique_count",
                "token_limit_count",
            ):
                metrics[f"steps_{step}_{mode}_{name}"] = cell[name]
    metrics.update(summary.get("runtime", {}))
    for name, value in summary.get("execution", {}).items():
        metrics[f"execution_{name}"] = value
    for pair, comparison in summary.get("paired_comparisons", {}).items():
        metrics[f"paired_comparisons_{pair}_accuracy_delta"] = comparison[
            "accuracy_delta"
        ]
    return metrics


def sample_measurement_tags(measurements):
    """Expose measured sample costs to the MLflow trace table's tag columns."""
    return {
        "peak_vram_gib": f"{measurements['peak_vram_bytes'] / 2**30:.3f}",
        "peak_reserved_vram_gib": (
            f"{measurements['peak_vram_reserved_bytes'] / 2**30:.3f}"
        ),
        "output_tokens_total": str(measurements["output_tokens_total"]),
        "latent_steps_total": str(measurements["latent_steps_total"]),
    }
