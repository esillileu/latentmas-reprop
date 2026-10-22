"""Plotting style configurations and constants for receiver compute preflight."""

from typing import Any

BUDGET_ORDER = ["64", "128", "256", "512", "1024", "free"]

CONDITION_STYLES: dict[str, dict[str, str]] = {
    "matched": {
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "-",
        "label": "Matched",
    },
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

PAIRING_COMPARISON_STYLES: dict[str, dict[str, str]] = {
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

TRADEOFF_COLOR_MAP: dict[tuple[str, int], str] = {
    ("matched", 10): "#72b0e0",
    ("matched", 20): "#1f77b4",
    ("mismatched", 10): "#ffb266",
    ("mismatched", 20): "#e66c00",
    ("no_handoff", 10): "#999999",
    ("no_handoff", 20): "#555555",
}


def get_budget_index(budget: Any) -> int:
    """Return the display index for a budget value."""
    b_str = str(budget)
    return BUDGET_ORDER.index(b_str) if b_str in BUDGET_ORDER else 999
