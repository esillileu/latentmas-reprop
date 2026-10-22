"""Visualization of receiver budget curves, handoff effects, and cost tradeoffs."""

from pathlib import Path
from typing import Any

from latentmas_reprop.application.receiver_compute_preflight.accuracy_plots import (
    plot_budget_accuracy_curve,
    plot_pairing_gain_curve,
)
from latentmas_reprop.application.receiver_compute_preflight.cost_plots import (
    plot_latency_accuracy_tradeoff,
    plot_tokens_cost_curve,
)
from latentmas_reprop.application.receiver_compute_preflight.plot_styles import (
    BUDGET_ORDER,
    CONDITION_STYLES,
    PAIRING_COMPARISON_STYLES,
    TRADEOFF_COLOR_MAP,
    get_budget_index,
)

# Backward-compatibility alias
_get_budget_index = get_budget_index


def generate_all_plots(
    directory: Path, metrics: dict[str, Any], model_name: str | None = None
) -> list[Path]:
    """Generate all standard diagnostic plots in the target directory."""
    directory.mkdir(parents=True, exist_ok=True)
    return [
        plot_budget_accuracy_curve(directory, metrics, model_name=model_name),
        plot_pairing_gain_curve(directory, metrics, model_name=model_name),
        plot_latency_accuracy_tradeoff(directory, metrics, model_name=model_name),
        plot_tokens_cost_curve(directory, metrics, model_name=model_name),
    ]


__all__ = [
    "BUDGET_ORDER",
    "CONDITION_STYLES",
    "PAIRING_COMPARISON_STYLES",
    "TRADEOFF_COLOR_MAP",
    "_get_budget_index",
    "generate_all_plots",
    "get_budget_index",
    "plot_budget_accuracy_curve",
    "plot_latency_accuracy_tradeoff",
    "plot_pairing_gain_curve",
    "plot_tokens_cost_curve",
]
