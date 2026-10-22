"""Latency, token, and attention cost curve visualizations for receiver compute preflight."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from latentmas_reprop.application.common.plotting import (
    add_threshold_lines,
    save_figure,
)
from latentmas_reprop.application.receiver_compute_preflight.plot_styles import (
    BUDGET_ORDER,
    CONDITION_STYLES,
    TRADEOFF_COLOR_MAP,
    get_budget_index,
)


def plot_latency_accuracy_tradeoff(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> Path:
    """Plot Receiver Latency vs Accuracy tradeoff."""
    curves = metrics.get("curves", [])
    if not curves:
        return directory / "latency_accuracy_tradeoff.png"

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    upstream_steps = sorted({c["upstream_steps"] for c in curves})
    marker_map = {"matched": "o", "mismatched": "s", "no_handoff": "^"}

    for cond in ("matched", "mismatched", "no_handoff"):
        for u in upstream_steps:
            sub = [
                c
                for c in curves
                if c["handoff_condition"] == cond and c["upstream_steps"] == u
            ]
            sub.sort(key=lambda c: get_budget_index(c["receiver_budget"]))

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
            col = TRADEOFF_COLOR_MAP.get((cond, u), "#333333")
            ls = (
                "--" if cond == "mismatched" else (":" if cond == "no_handoff" else "-")
            )

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

    add_threshold_lines(ax, [0.5, 0.7, 0.8, 0.9], with_labels=False)

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

    return save_figure(fig, directory / "latency_accuracy_tradeoff.png")


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
            sub.sort(key=lambda c: get_budget_index(c["receiver_budget"]))
            x = [
                get_budget_index(c["receiver_budget"])
                for c in sub
                if c.get("mean_generated_tokens") is not None
            ]
            y = [
                c["mean_generated_tokens"]
                for c in sub
                if c.get("mean_generated_tokens") is not None
            ]
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
            sub.sort(key=lambda c: get_budget_index(c["receiver_budget"]))
            x = [
                get_budget_index(c["receiver_budget"])
                for c in sub
                if c.get("mean_receiver_attention_pairs") is not None
            ]
            y = [
                c["mean_receiver_attention_pairs"]
                for c in sub
                if c.get("mean_receiver_attention_pairs") is not None
            ]
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

    return save_figure(fig, directory / "tokens_cost_curve.png")
