"""Tables calculated from saved MLflow observations and probe permutation arrays."""

import csv
import json
from pathlib import Path

import numpy as np

from .inference import correct_receiver_families


def sender_cells(results, statistics):
    """Recompute probe effects and max-statistic p-values within one saved run."""
    order = statistics["cell_order"]
    null = np.asarray(statistics["permutation_accuracies"], dtype=float)
    observed = {(c["representation"], c["step"]): c for c in results["cells"]}
    if null.size and (null.ndim != 2 or null.shape[1] != len(order)):
        raise ValueError("Probe permutation array does not match cell order")
    cells = []
    for index, spec in enumerate(order):
        accuracy = observed[(spec["representation"], spec["step"])][
            "observed_oof_accuracy"
        ]
        distribution = null[:, index] if null.size else np.array([])
        raw = (
            (1 + np.count_nonzero(distribution >= accuracy)) / (1 + len(null))
            if null.size
            else None
        )
        fwer = (
            (1 + np.count_nonzero(null.max(axis=1) >= accuracy)) / (1 + len(null))
            if null.size
            else None
        )
        statistic = float(distribution.mean()) if distribution.size else None
        cells.append(
            {
                **spec,
                "probe_accuracy": accuracy,
                "null_mean_accuracy": statistic,
                "observed_minus_null": accuracy - statistic
                if statistic is not None
                else None,
                "raw_p_value": raw,
                "fwer_p_value": fwer,
                "significance": bool(fwer < 0.05) if fwer is not None else None,
            }
        )
    return cells


def write_table(output: Path, name: str, rows):
    (output / f"{name}.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / f"{name}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def export_tables(output, runs, probes, robustness=(), geometry=()):
    correct_receiver_families(runs)
    conditions, digits, dependencies, summaries, tests, samples = [], [], [], [], [], []
    for run in runs:
        identity = {key: run[key] for key in ("model", "latent_steps", "run_id")}
        summary = dict(identity)
        for condition, values in run["conditions"].items():
            base = {**identity, "condition": condition}
            conditions.append(
                {
                    **base,
                    **{
                        k: v
                        for k, v in values.items()
                        if k
                        not in (
                            "dependency",
                            "per_digit_accuracy",
                            "probability_samples",
                            "soft_dependency",
                            "exact_recovery",
                        )
                    },
                }
            )
            for digit, stat in enumerate(values["per_digit_accuracy"]):
                digits.append(
                    {
                        **base,
                        "digit": digit,
                        **{
                            k: v
                            for k, v in stat.items()
                            if k.startswith("mean_") or k == "probability_sample_count"
                        },
                        "source_count": stat["sample_count"],
                        "correct_count": stat["correct_count"],
                        "recall": stat["accuracy"],
                        "predicted_count": values["prediction_distribution"][
                            str(digit)
                        ]["count"],
                    }
                )
            if "dependency" in values:
                dependencies.append({**base, **values["dependency"]})
            samples.extend(
                {**base, **row} for row in values.get("probability_samples", [])
            )
            condition_tests = {
                "balanced_accuracy": values.get("exact_recovery"),
                **values.get("soft_dependency", {}),
            }
            for metric, test in condition_tests.items():
                tests.append(
                    {
                        **base,
                        "metric": metric,
                        **(test or {"observed": None, "raw_p_value": None}),
                    }
                )
            for metric in (
                "sample_count",
                "accuracy",
                "balanced_accuracy",
                "matched_drop_accuracy",
                "correct_class_probability",
                "probability_delta_vs_drop",
                "argmax_changed_fraction",
                "source_digit_counts",
            ):
                summary[f"{condition}/{metric}"] = values[metric]
        summaries.append(summary)
    for name, rows in (
        ("sender_probe_cells", probes),
        ("receiver_conditions", conditions),
        ("receiver_per_digit", digits),
        ("receiver_dependencies", dependencies),
        ("model_steps_summary", summaries),
        ("receiver_inference_tests", tests),
        ("receiver_probability_samples", samples),
        ("sender_robustness", robustness),
        ("sender_geometry", geometry),
    ):
        write_table(output, name, rows)
