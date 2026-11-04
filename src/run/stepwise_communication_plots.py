"""Visualize sender and receiver communication metrics."""

import math

from .presentation_plots import STYLE, plt, save
from .stepwise_communication_metrics import models_in

COLORS = ("#7B3294", "#0072B2", "#D55E00", "#009E73")


def plot_metrics(rows, output, annotate=False):
    models = models_in(rows)
    colors = COLORS[: len(models)]
    effects = [row["probe_effect_pp"] for row in rows]
    signal_start = 2 * math.floor(min(0, min(effects)) / 2)
    signal_end = 2 * math.ceil(max(0, max(effects)) / 2)
    signal_ticks = range(signal_start, max(signal_start + 2, signal_end) + 1, 2)
    signal_limits = (signal_ticks[0] - 0.5, signal_ticks[-1] + 0.5)
    receiver_limits = (-3, 103)
    receiver_ticks = range(0, 101, 20)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for model, color in zip(models, colors, strict=True):
            cells = [row for row in rows if row["model"] == model]
            xs = [row["probe_effect_pp"] for row in cells]
            ys = [row["receiver_changed_pct"] for row in cells]
            ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.5)
            for significant in (True, False):
                subset = [
                    row for row in cells if row["probe_significant"] == significant
                ]
                ax.scatter(
                    [row["probe_effect_pp"] for row in subset],
                    [row["receiver_changed_pct"] for row in subset],
                    s=110,
                    edgecolors=color,
                    facecolors=color if significant else "none",
                    linewidths=1.8,
                    zorder=3,
                )
            ax.plot([], [], "o", color=color, label=model.split("-")[-1])
            if annotate:
                for row in cells:
                    if row["latent_step"] in (1, 4, 20):
                        ax.annotate(
                            str(row["latent_step"]),
                            (row["probe_effect_pp"], row["receiver_changed_pct"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=11,
                        )
        ax.set(
            xlabel="Digit signal in handoff latent (pp above null)",
            ylabel="Receiver predictions changed by handoff (%)",
            xlim=signal_limits,
            xticks=signal_ticks,
            ylim=receiver_limits,
            yticks=receiver_ticks,
        )
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(16)
        ax.grid(False)
        ax.legend(frameon=False, ncol=len(models), fontsize=14)
        save(fig, output, "stepwise_communication_scatter")
    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(4.7 * len(models), 7),
            layout="constrained",
            sharex=True,
            sharey="row",
        )
        for column, (model, color) in enumerate(zip(models, colors, strict=True)):
            cells = [row for row in rows if row["model"] == model]
            for index, field in enumerate(("probe_effect_pp", "receiver_changed_pct")):
                ax = axes[index, column]
                ax.plot(
                    [row["latent_step"] for row in cells],
                    [row[field] for row in cells],
                    "o-",
                    color=color,
                    linewidth=1,
                )
                ax.set(
                    title=model,
                    xlabel="latent_step",
                    ylabel=field,
                    xlim=(0.5, 20.5),
                    xticks=(1, 4, 10, 15, 20),
                    ylim=signal_limits if index == 0 else receiver_limits,
                    yticks=signal_ticks if index == 0 else receiver_ticks,
                )
                ax.grid(alpha=0.25)
        save(fig, output, "stepwise_communication_diagnostics")
