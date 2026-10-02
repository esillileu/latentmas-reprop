"""Visualization of receiver budget curves, handoff effects, and cost tradeoffs."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUDGET_ORDER = ["64", "128", "256", "512", "1024", "free"]
CONDITION_STYLES = {
    "matched": {"color": "#1f77b4", "marker": "o", "linestyle": "-", "label": "Matched"},
    "mismatched": {
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "--",
        "label": "Mismatched",
    },
    "no_handoff": {
        "color": "#7f7f7f",
        "marker": "^",
        "linestyle": ":",
        "label": "No Handoff",
    },
}


def _get_budget_index(budget: Any) -> int:
    b_str = str(budget)
    return BUDGET_ORDER.index(b_str) if b_str in BUDGET_ORDER else 999


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
            cond_curves.sort(key=lambda c: _get_budget_index(c["receiver_budget"]))

            x_indices, y_vals, y_err_low, y_err_high = [], [], [], []
            for c in cond_curves:
                acc = c.get("accuracy")
                if acc is not None and not np.isnan(acc):
                    x_idx = _get_budget_index(c["receiver_budget"])
                    x_indices.append(x_idx)
                    y_vals.append(acc)
                    low = c.get("ci_low", acc)
                    high = c.get("ci_high", acc)
                    y_err_low.append(max(0.0, acc - (low if low is not None else acc)))
                    y_err_high.append(
                        max(0.0, (high if high is not None else acc) - acc)
                    )

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

        # Target threshold reference lines
        for tau in [0.5, 0.7, 0.8, 0.9]:
            ax.axhline(tau, color="#cccccc", linestyle=":", linewidth=0.9, alpha=0.8)
            ax.text(
                -0.45,
                tau + 0.015,
                f"τ={tau}",
                color="#888888",
                fontsize=8,
                verticalalignment="bottom",
            )

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
    fig.tight_layout()

    out_path = directory / "budget_accuracy_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pairing_gain_curve(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Causal Handoff Pairing Gain and Baselines across budgets."""
    comparisons = metrics.get("comparisons", [])
    if not comparisons:
        return directory / "pairing_gain_curve.png"

    upstream_steps = sorted({c["upstream_steps"] for c in comparisons})
    fig, axes = plt.subplots(
        1, len(upstream_steps), figsize=(6.5 * len(upstream_steps), 5.0), sharey=True, squeeze=False
    )
    axes_flat = axes[0]

    comp_styles = {
        "pairing_gain": {
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-",
            "label": "Pairing Gain (Matched - Mismatched)",
        },
        "matched_vs_no_handoff": {
            "color": "#2ca02c",
            "marker": "s",
            "linestyle": "--",
            "label": "Matched - No Handoff",
        },
        "mismatched_vs_no_handoff": {
            "color": "#d62728",
            "marker": "^",
            "linestyle": ":",
            "label": "Mismatched - No Handoff",
        },
    }

    for idx, u in enumerate(upstream_steps):
        ax = axes_flat[idx]
        u_comps = [c for c in comparisons if c["upstream_steps"] == u]

        for comp_name, style in comp_styles.items():
            c_rows = [c for c in u_comps if c["comparison"] == comp_name]
            c_rows.sort(key=lambda c: _get_budget_index(c["receiver_budget"]))

            x_idx, y_vals, y_err_low, y_err_high = [], [], [], []
            for r in c_rows:
                diff = r.get("difference")
                if diff is not None and not np.isnan(diff):
                    x_idx.append(_get_budget_index(r["receiver_budget"]))
                    y_vals.append(diff)
                    low = r.get("ci_low", diff)
                    high = r.get("ci_high", diff)
                    y_err_low.append(max(0.0, diff - (low if low is not None else diff)))
                    y_err_high.append(
                        max(0.0, (high if high is not None else diff) - diff)
                    )

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

        ax.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0, alpha=0.7)
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
    fig.tight_layout()

    out_path = directory / "pairing_gain_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_latency_accuracy_tradeoff(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Receiver Latency vs Accuracy tradeoff."""
    curves = metrics.get("curves", [])
    if not curves:
        return directory / "latency_accuracy_tradeoff.png"

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    upstream_steps = sorted({c["upstream_steps"] for c in curves})

    color_map = {
        ("matched", 10): "#72b0e0",
        ("matched", 20): "#1f77b4",
        ("mismatched", 10): "#ffb266",
        ("mismatched", 20): "#e66c00",
        ("no_handoff", 10): "#999999",
        ("no_handoff", 20): "#555555",
    }
    marker_map = {"matched": "o", "mismatched": "s", "no_handoff": "^"}

    for cond in ("matched", "mismatched", "no_handoff"):
        for u in upstream_steps:
            sub = [
                c
                for c in curves
                if c["handoff_condition"] == cond and c["upstream_steps"] == u
            ]
            sub.sort(key=lambda c: _get_budget_index(c["receiver_budget"]))

            latencies = [c.get("mean_receiver_latency_sec") for c in sub]
            accuracies = [c.get("accuracy") for c in sub]

            valid_pts = [
                (lat, acc, str(c["receiver_budget"]))
                for lat, acc, c in zip(latencies, accuracies, sub, strict=True)
                if lat is not None and acc is not None and not np.isnan(acc)
            ]
            if not valid_pts:
                continue

            x = [p[0] for p in valid_pts]
            y = [p[1] for p in valid_pts]
            col = color_map.get((cond, u), "#333333")
            ls = "--" if cond == "mismatched" else (":" if cond == "no_handoff" else "-")

            ax.plot(
                x,
                y,
                color=col,
                linestyle=ls,
                marker=marker_map.get(cond, "o"),
                linewidth=2.0,
                markersize=6,
                label=f"{cond.capitalize()} (U={u})",
            )
            for lx, ly, b in valid_pts:
                if b in ("512", "1024", "free"):
                    ax.annotate(
                        f"R={b}",
                        (lx, ly),
                        textcoords="offset points",
                        xytext=(5, -2),
                        fontsize=8,
                        alpha=0.8,
                    )

    for tau in [0.5, 0.7, 0.8, 0.9]:
        ax.axhline(tau, color="#cccccc", linestyle=":", linewidth=0.9, alpha=0.8)

    ax.set_xlabel("Mean Receiver Latency (seconds)", fontsize=11)
    ax.set_ylabel("Final-Answer Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    title_model = f" ({model_name})" if model_name else ""
    ax.set_title(
        f"Receiver Latency vs Accuracy Tradeoff{title_model}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out_path = directory / "latency_accuracy_tradeoff.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tokens_cost_curve(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Mean Generated Tokens and Attention Pairs across budgets."""
    curves = metrics.get("curves", [])
    if not curves:
        return directory / "tokens_cost_curve.png"

    upstream_steps = sorted({c["upstream_steps"] for c in curves})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # Panel 1: Generated Tokens
    ax1 = axes[0]
    for cond, style in CONDITION_STYLES.items():
        for u in upstream_steps:
            sub = [
                c
                for c in curves
                if c["handoff_condition"] == cond and c["upstream_steps"] == u
            ]
            sub.sort(key=lambda c: _get_budget_index(c["receiver_budget"]))
            x = [_get_budget_index(c["receiver_budget"]) for c in sub if c.get("mean_generated_tokens") is not None]
            y = [c["mean_generated_tokens"] for c in sub if c.get("mean_generated_tokens") is not None]
            alpha = 1.0 if u == max(upstream_steps) else 0.5
            if x:
                ax1.plot(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    alpha=alpha,
                    label=f"{style['label']} (U={u})",
                )

    ax1.set_title("Mean Generated Tokens", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Receiver Budget (R)", fontsize=11)
    ax1.set_ylabel("Generated Tokens (EOS included)", fontsize=11)
    ax1.set_xticks(range(len(BUDGET_ORDER)))
    ax1.set_xticklabels(BUDGET_ORDER)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Panel 2: Attention Pairs Proxy
    ax2 = axes[1]
    for cond, style in CONDITION_STYLES.items():
        for u in upstream_steps:
            sub = [
                c
                for c in curves
                if c["handoff_condition"] == cond and c["upstream_steps"] == u
            ]
            sub.sort(key=lambda c: _get_budget_index(c["receiver_budget"]))
            x = [_get_budget_index(c["receiver_budget"]) for c in sub if c.get("mean_receiver_attention_pairs") is not None]
            y = [c["mean_receiver_attention_pairs"] for c in sub if c.get("mean_receiver_attention_pairs") is not None]
            alpha = 1.0 if u == max(upstream_steps) else 0.5
            if x:
                ax2.plot(
                    x,
                    y,
                    color=style["color"],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=1.8,
                    alpha=alpha,
                    label=f"{style['label']} (U={u})",
                )

    ax2.set_title("Mean Receiver Attention Pairs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Receiver Budget (R)", fontsize=11)
    ax2.set_ylabel("Attention Pairs (Causal Q*K)", fontsize=11)
    ax2.set_xticks(range(len(BUDGET_ORDER)))
    ax2.set_xticklabels(BUDGET_ORDER)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)

    title_model = f" ({model_name})" if model_name else ""
    fig.suptitle(
        f"Receiver Computational Cost Curves{title_model}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_path = directory / "tokens_cost_curve.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_plots(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> list[Path]:
    """Generate all standard diagnostic plots in the target directory."""
    directory.mkdir(parents=True, exist_ok=True)
    generated = [
        plot_budget_accuracy_curve(directory, metrics, model_name=model_name),
        plot_pairing_gain_curve(directory, metrics, model_name=model_name),
        plot_latency_accuracy_tradeoff(directory, metrics, model_name=model_name),
        plot_tokens_cost_curve(directory, metrics, model_name=model_name),
    ]
    return generated
