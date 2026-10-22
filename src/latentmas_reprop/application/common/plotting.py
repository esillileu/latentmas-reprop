"""Common plotting utilities and figure helpers."""

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt


def save_figure(
    fig: matplotlib.figure.Figure,
    out_path: Path,
    dpi: int = 200,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure to disk and optionally close it."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return out_path


def compute_asymmetric_error(
    value: float,
    ci_low: float | None = None,
    ci_high: float | None = None,
) -> tuple[float, float]:
    """Compute asymmetric lower and upper error margins for error bars."""
    low = value if ci_low is None else ci_low
    high = value if ci_high is None else ci_high
    return max(0.0, value - low), max(0.0, high - value)


def add_threshold_lines(
    ax: matplotlib.axes.Axes,
    thresholds: Sequence[float] = (0.5, 0.7, 0.8, 0.9),
    with_labels: bool = True,
    x_pos: float = -0.45,
) -> None:
    """Draw horizontal target threshold reference lines."""
    for tau in thresholds:
        ax.axhline(tau, color="#cccccc", linestyle=":", linewidth=0.9, alpha=0.8)
        if with_labels:
            ax.text(
                x_pos,
                tau + 0.015,
                f"τ={tau}",
                color="#888888",
                fontsize=8,
                verticalalignment="bottom",
            )


def add_zero_line(
    ax: matplotlib.axes.Axes,
    color: str = "#333333",
    linestyle: str = "-",
    linewidth: float = 1.0,
    alpha: float = 0.7,
) -> None:
    """Draw a horizontal reference line at y=0."""
    ax.axhline(0.0, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
