"""Accuracy and causal pairing gain curve visualizations for receiver compute preflight."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from latentmas_reprop.application.common.plotting import (
    add_threshold_lines,
    add_zero_line,
    compute_asymmetric_error,
    save_figure,
)
from latentmas_reprop.application.receiver_compute_preflight.plot_styles import (
    BUDGET_ORDER,
    CONDITION_STYLES,
    PAIRING_COMPARISON_STYLES,
    get_budget_index,
)


def plot_budget_accuracy_curve(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Receiver Budget vs Accuracy Curve with 95% bootstrap CI."""
    curves = metrics.get("curves", [])
    if not curves:
        return directory / "budget_accuracy_curve.png"

    upstream_steps = sorted({c["upstream_steps"] for c in curves})
    num_panels = len(upstream_steps)

    fig, axes = plt.subplots(
        1, num_panels, figsize=(6.5 * num_panels, 5.0), sharey=True, squeeze=False
    )
    axes_flat = axes[0]

    for idx, u in enumerate(upstream_steps):
        ax = axes_flat[idx]
        u_curves = [c for c in curves if c["upstream_steps"] == u]

        for cond, style in CONDITION_STYLES.items():
            cond_curves = [c for c in u_curves if c["handoff_condition"] == cond]
            cond_curves.sort(key=lambda c: get_budget_index(c["receiver_budget"]))

            x_indices, y_vals, y_err_low, y_err_high = [], [], [], []
            for c in cond_curves:
                acc = c.get("accuracy")
                if acc is not None and not np.isnan(acc):
                    x_idx = get_budget_index(c["receiver_budget"])
                    x_indices.append(x_idx)
                    y_vals.append(acc)
                    err_low, err_high = compute_asymmetric_error(
                        acc, c.get("ci_low"), c.get("ci_high")
                    )
                    y_err_low.append(err_low)
                    y_err_high.append(err_high)

            if x_indices:
                ax.errorbar(
                    x_indices,
                    y_vals,
                    yerr=[y_err_low, y_err_high],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    markersize=6,
                    capsize=4,
                    label=style["label"],
                )

        add_threshold_lines(ax, [0.5, 0.7, 0.8, 0.9])

        ax.set_title(f"Upstream Steps U = {u}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Receiver Token Budget (R)", fontsize=11)
        ax.set_xticks(range(len(BUDGET_ORDER)))
        ax.set_xticklabels(BUDGET_ORDER)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Final-Answer Accuracy", fontsize=11)
            ax.legend(loc="upper left", framealpha=0.9)

    title_model = f" ({model_name})" if model_name else ""
    fig.suptitle(
        f"Receiver-Budget Accuracy Curves{title_model}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    return save_figure(fig, directory / "budget_accuracy_curve.png")


def plot_pairing_gain_curve(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Causal Handoff Pairing Gain and Baselines across budgets."""
    comparisons = metrics.get("comparisons", [])
    if not comparisons:
        return directory / "pairing_gain_curve.png"

    upstream_steps = sorted({c["upstream_steps"] for c in comparisons})
    fig, axes = plt.subplots(
        1,
        len(upstream_steps),
        figsize=(6.5 * len(upstream_steps), 5.0),
        sharey=True,
        squeeze=False,
    )
    axes_flat = axes[0]

    for idx, u in enumerate(upstream_steps):
        ax = axes_flat[idx]
        u_comps = [c for c in comparisons if c["upstream_steps"] == u]

        for comp_name, style in PAIRING_COMPARISON_STYLES.items():
            c_rows = [c for c in u_comps if c["comparison"] == comp_name]
            c_rows.sort(key=lambda c: get_budget_index(c["receiver_budget"]))

            x_idx, y_vals, y_err_low, y_err_high = [], [], [], []
            for r in c_rows:
                diff = r.get("difference")
                if diff is not None and not np.isnan(diff):
                    x_idx.append(get_budget_index(r["receiver_budget"]))
                    y_vals.append(diff)
                    err_low, err_high = compute_asymmetric_error(
                        diff, r.get("ci_low"), r.get("ci_high")
                    )
                    y_err_low.append(err_low)
                    y_err_high.append(err_high)

            if x_idx:
                ax.errorbar(
                    x_idx,
                    y_vals,
                    yerr=[y_err_low, y_err_high],
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2.0,
                    markersize=6,
                    capsize=4,
                    label=style["label"],
                )

        add_zero_line(ax)
        ax.set_title(f"Upstream Steps U = {u}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Receiver Token Budget (R)", fontsize=11)
        ax.set_xticks(range(len(BUDGET_ORDER)))
        ax.set_xticklabels(BUDGET_ORDER)
        ax.grid(True, linestyle="--", alpha=0.5)
        if idx == 0:
            ax.set_ylabel("Accuracy Difference (Paired 95% CI)", fontsize=11)
            ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    title_model = f" ({model_name})" if model_name else ""
    fig.suptitle(
        f"Causal Handoff Effects & Pairing Gain{title_model}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    return save_figure(fig, directory / "pairing_gain_curve.png")
