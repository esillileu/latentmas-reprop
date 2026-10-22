"""Presentation scatter comparing secret-digit signal and receiver influence."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def create_digit_aspects_figure():
    """Plot the nine user-supplied secret-digit cells without recomputing metrics."""
    data = (
        (
            "0.6B",
            "#7B3294",
            (8.47, 7.02, 7.00),
            (0, 100, 0),
            ((12, 16), (-14, -24), (12, 16)),
        ),
        (
            "4B",
            "#0072B2",
            (6.51, 0.02, 3.99),
            (0, 80, 100),
            ((-16, 16), (16, 0), (0, -24)),
        ),
        (
            "8B",
            "#D55E00",
            (9.51, 7.49, 8.99),
            (70, 100, 100),
            ((-16, 0), (14, -24), (12, -24)),
        ),
    )
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    for model, color, xs, ys, offsets in data:
        ax.plot(xs, ys, color=color, linewidth=1.3, alpha=0.65, zorder=2)
        for step, x, y, offset in zip((1, 4, 20), xs, ys, offsets, strict=True):
            ax.plot(
                x,
                y,
                marker="o",
                markersize=14,
                linestyle="none",
                markerfacecolor="white" if model == "4B" and step in (4, 20) else color,
                markeredgecolor=color,
                markeredgewidth=2.5,
                clip_on=False,
                zorder=3,
            )
            ax.annotate(
                str(step),
                (x, y),
                xytext=offset,
                textcoords="offset points",
                color=color,
                fontsize=22,
                fontweight="bold",
                ha="left" if offset[0] > 0 else "right" if offset[0] < 0 else "center",
                va="center",
                annotation_clip=False,
            )
    ax.set(
        xlabel="Digit signal detectable in handoff latent\n(pp above null)",
        ylabel="Receiver predictions changed by handoff (%)",
        xlim=(0, 10),
        ylim=(0, 100),
    )
    ax.set_title("Different aspects of “communication” do not move together", pad=65)
    ax.set_xticks(range(0, 11, 2))
    ax.set_yticks(range(0, 101, 20))
    ax.grid(False)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=color,
            markersize=12,
            label=model,
        )
        for model, color, *_ in data
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=3,
        prop={"size": 22, "weight": "bold"},
        handletextpad=0.4,
        columnspacing=1,
    )
    return fig
