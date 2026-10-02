"""Human-readable saved-observation reports."""

import json
from typing import Any

from .inference import correct_receiver_families


def sender_markdown(probes):
    lines = []
    for steps, title in (
        (20, "Sender main trajectory: 20-step runs"),
        (None, "Sender comparison: 1-step / 4-step runs"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "Null statistic is mean permutation accuracy. FWER uses the maximum accuracy across all steps and both representations in the same run; significance uses FWER p < 0.05. Effect CI: 95% template-cluster percentile bootstrap with fixed OOF predictions and fixed null mean; no probe refitting.",
            "",
            "| Model | Run ID | Run steps | Representation | Step | Accuracy | Null | Effect | Effect CI low | Effect CI high | Raw p | FWER p | Significant |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for row in probes:
            if (steps == 20 and row["latent_steps"] != 20) or (
                steps is None and row["latent_steps"] not in (1, 4)
            ):
                continue
            values = [
                row.get(k)
                for k in (
                    "model",
                    "run_id",
                    "latent_steps",
                    "representation",
                    "step",
                    "probe_accuracy",
                    "null_mean_accuracy",
                    "observed_minus_null",
                    "effect_ci_low",
                    "effect_ci_high",
                    "raw_p_value",
                    "fwer_p_value",
                    "significance",
                )
            ]
            lines.append(
                "| "
                + " | ".join(
                    "—" if v is None else f"{v:.6f}" if isinstance(v, float) else str(v)
                    for v in values
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def render_markdown(
    runs: list[dict[str, Any]], probes=(), robustness=(), geometry=()
) -> str:
    correct_receiver_families(runs)

    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = [
        "# Saved preflight statistics",
        "",
        "No inference was run. Entropy, MI and NLL use natural logarithms. NLL and Brier use normalized digit probabilities. Balanced accuracy averages recall over represented source digits.",
        "",
        "| Model | Steps | Run ID | Samples |",
        "|---|---:|---|---:|",
    ]
    for run in runs:
        lines.append(
            f"| {run['model']} | {run['latent_steps']} | {run['run_id']} | {run['sample_count']} |"
        )
    permutation_settings = sorted(
        {
            (str(dep.get("permutations", "—")), str(dep.get("seed", "—")))
            for run in runs
            for values in run["conditions"].values()
            if (dep := values.get("dependency")) is not None
        }
    )
    settings = (
        "; ".join(
            f"permutation count: {count}, seed: {seed}"
            for count, seed in permutation_settings
        )
        or "permutation count: —, seed: —"
    )
    lines += [
        "",
        f"MI p-value settings: {settings}. Shuffle unit: individual sample source labels within each run and condition; predictions remain fixed.",
        "",
        "| Model | Steps | Condition | N | Raw accuracy | Balanced accuracy | Matched Drop | Δ P(correct) | Changed | Prediction entropy | Digit mass Δ vs drop | JS divergence vs drop | NLL Δ vs drop | Brier Δ vs drop | MI | MI p | MI FWER p (Holm) | Source counts (0-9) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    inference_lines = [
        "",
        "## Receiver permutation tests and probability observations",
        "",
        "MI Holm FWER family: all reported Receiver Own/Cross conditions across models and steps. NLL, Brier and balanced accuracy each use a separate Own/Cross Holm family.",
        "",
        f"Loss/recovery permutation settings: {settings}. RNG streams: 1 for balanced accuracy, 2 for NLL/Brier (shared loss shuffles).",
        "",
        "Source labels are shuffled within run and condition; probabilities and predictions stay fixed. NLL/Brier use the lower tail; balanced accuracy uses the upper tail. Rank ties use ascending digit order. Top-k columns are sample fractions.",
        "",
        "| Model | Steps | Condition | Balanced accuracy p | Balanced accuracy Holm p | NLL observed | NLL null mean | NLL p | NLL Holm p | Brier observed | Brier null mean | Brier p | Brier Holm p | Mean rank | Top-2 | Top-3 | Mean margin |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        for condition, values in run["conditions"].items():
            recovery = values.get("exact_recovery", {})
            soft = values.get("soft_dependency", {})
            nll, brier = (
                soft.get("source_label_nll") or {},
                soft.get("brier_score") or {},
            )
            cells = [
                recovery.get("raw_p_value"),
                recovery.get("holm_p_value"),
                nll.get("observed"),
                nll.get("null_mean"),
                nll.get("raw_p_value"),
                nll.get("holm_p_value"),
                brier.get("observed"),
                brier.get("null_mean"),
                brier.get("raw_p_value"),
                brier.get("holm_p_value"),
                values.get("mean_correct_digit_rank"),
                values.get("mean_top_2"),
                values.get("mean_top_3"),
                values.get("mean_correct_digit_margin"),
            ]
            inference_lines.append(
                f"| {run['model']} | {run['latent_steps']} | {condition} | "
                + " | ".join(fmt(v) for v in cells)
                + " |"
            )
    details = []
    for run in runs:
        for condition, v in run["conditions"].items():
            dep = v.get("dependency", {})
            lines.append(
                f"| {run['model']} | {run['latent_steps']} | {condition} | {v['sample_count']} | {fmt(v['accuracy'])} | {fmt(v['balanced_accuracy'])} | {fmt(v['matched_drop_accuracy'])} | {fmt(v['probability_delta_vs_drop'])} | {fmt(v['argmax_changed_fraction'])} | {fmt(v['prediction_entropy'])} | {fmt(v['digit_probability_mass_delta_vs_drop'])} | {fmt(v['js_divergence_vs_drop'])} | {fmt(v['source_label_nll_delta_vs_drop'])} | {fmt(v['brier_score_delta_vs_drop'])} | {fmt(dep.get('mutual_information'))} | {fmt(dep.get('permutation_p_value'))} | {fmt(dep.get('holm_p_value'))} | {v['source_digit_counts']} |"
            )
            details += [
                "",
                f"<details><summary>{run['model']} / {run['latent_steps']} / {condition}: statistics and source x prediction</summary>",
                "",
                "```json",
                json.dumps(
                    {
                        k: x
                        for k, x in v.items()
                        if k
                        not in (
                            "confusion_matrix",
                            "dependency",
                            "probability_samples",
                            "soft_dependency",
                            "exact_recovery",
                        )
                    },
                    indent=2,
                ),
                "```",
                "",
                "| Source | "
                + " | ".join(map(str, range(10)))
                + " | Recall | Mean P(correct) | Mean q(correct) | NLL | Brier |",
                "|---:|" + "---:|" * 15,
            ]
            for digit, row in enumerate(v["confusion_matrix"]):
                details.append(
                    f"| {digit} | "
                    + " | ".join(map(str, row))
                    + " | "
                    + " | ".join(
                        fmt(v["per_digit_accuracy"][digit].get(k))
                        for k in (
                            "accuracy",
                            "mean_correct_probability",
                            "mean_normalized_correct_probability",
                            "mean_source_label_nll",
                            "mean_brier_score",
                        )
                    )
                    + " |"
                )
            details += ["", "</details>", ""]
    return (
        "\n".join(
            lines
            + inference_lines
            + (sender_markdown(probes).splitlines() if probes else [])
            + sender_diagnostics_markdown(robustness, geometry)
            + details
        )
        + "\n"
    )


def sender_diagnostics_markdown(robustness, geometry):
    def fmt(value):
        return "—" if value is None else f"{value:.4f}"

    lines = []
    for title, rows, keys in (
        (
            "Sender probe robustness",
            robustness,
            ("probe", "standardization", "parameter", "accuracy"),
        ),
        (
            "Sender geometry",
            geometry,
            (
                "centroid_separation_mean",
                "within_class_scatter",
                "between_class_scatter",
                "between_within_ratio",
                "effective_rank",
            ),
        ),
    ):
        lines += [
            "",
            f"<details><summary>{title}</summary>",
            "",
            "Saved raw features only. Robustness uses saved template-grouped folds; standardization is fitted on each training fold. All preset probes are reported without choosing a best configuration. Geometry uses unstandardized features; covariance is sample-centered with denominator n-1. Full spectra and settings are saved in CSV/JSON.",
            "",
            "| Model | Steps | Representation | Step | " + " | ".join(keys) + " |",
            "|---|---:|---|---:|" + "---:|" * len(keys),
        ]
        for row in rows:
            cells = [
                str(row.get(k, "—"))
                if k in ("probe", "standardization", "parameter")
                else fmt(row.get(k))
                for k in keys
            ]
            lines.append(
                f"| {row['model']} | {row['latent_steps']} | {row['representation']} | {row['step']} | "
                + " | ".join(cells)
                + " |"
            )
        if not rows:
            lines += ["", "—"]
        lines += ["", "</details>", ""]
    return lines
