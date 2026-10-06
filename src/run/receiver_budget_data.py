"""Saved presentation data grids and discrete receiver-budget thresholds."""

import csv
from pathlib import Path

TARGET = 0.50


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def unique_grid(rows, key, expected, source):
    """Refuse missing/duplicate cells instead of pooling distinct runs."""
    cells = {}
    for row in rows:
        cell = key(row)
        if cell in cells:
            raise ValueError(f"Duplicate cell {cell!r} in {source}")
        cells[cell] = row
    if set(cells) != set(expected):
        raise ValueError(f"Expected cells {expected!r} in {source}; got {list(cells)}")
    return cells


def percent(row, field):
    value = float(row[field])
    if not 0 <= value <= 1:
        raise ValueError(f"Invalid {field}: {row[field]!r}")
    return 100 * value


def budget_value(value):
    return "free" if str(value).strip() == "free" else int(value)


def budget_order(value):
    return (value == "free", 0 if value == "free" else int(value))


def minimum_budget(rows, target=TARGET):
    return min(
        (
            budget_value(r["receiver_budget"])
            for r in rows
            if r["receiver_budget"] != "free" and float(r["accuracy"]) >= target
        ),
        default=None,
    )
