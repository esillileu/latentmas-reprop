"""Validate and compare sender and receiver metrics."""

import math

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B")


def number(row, field):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {field}: {row}") from exc
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Expected finite fraction for {field}: {row}")
    return value


def models_in(rows):
    return MODELS if any(row["model"] == MODELS[-1] for row in rows) else MODELS[:3]


def grid(rows, step_field, steps, source):
    cells = {}
    expected = {(model, step) for model in models_in(rows) for step in steps}
    for row in rows:
        try:
            key = (row["model"], int(row[step_field]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid {source} key: {row}") from exc
        if key in cells:
            raise ValueError(f"Duplicate {source} key: {key}")
        cells[key] = row
    if set(cells) != expected or len(rows) != len(expected):
        raise ValueError(
            f"{source}: expected {len(expected)} unique rows; "
            f"missing={sorted(expected - cells.keys())}, "
            f"unexpected={sorted(cells.keys() - expected)}"
        )
    return cells


def communication_metrics(probes, sweep):
    selected = [
        row
        for row in probes
        if row.get("latent_steps") == "20"
        and row.get("representation") == "latent_post_realign"
    ]
    probe_grid = grid(selected, "step", range(1, 21), "sender probe")
    sweep_grid = grid(sweep, "latent_step", range(1, 21), "receiver sweep")
    rows = []
    for key, probe in probe_grid.items():
        accuracy = number(probe, "probe_accuracy")
        null = number(probe, "null_mean_accuracy")
        p_value = number(probe, "fwer_p_value")
        changed = number(sweep_grid[key], "receiver_changed_fraction")
        rows.append(
            {
                "model": key[0],
                "latent_step": key[1],
                "probe_accuracy": accuracy,
                "probe_null_mean": null,
                "probe_effect_pp": 100 * (accuracy - null),
                "probe_fwer_p": p_value,
                "probe_significant": p_value < 0.05,
                "receiver_changed_fraction": changed,
                "receiver_changed_pct": 100 * changed,
            }
        )
    return sorted(
        rows, key=lambda row: (MODELS.index(row["model"]), row["latent_step"])
    )


def receiver_parity(metrics, independent):
    """Compare saved latent-only own change fractions at steps 1, 4 and 20."""
    selected = [row for row in independent if row.get("condition") == "latent_only/own"]
    cells = grid(selected, "latent_steps", (1, 4, 20), "independent receiver")
    result = []
    for row in metrics:
        key = (row["model"], row["latent_step"])
        if key[1] not in (1, 4, 20):
            continue
        original = (
            number(cells[key], "argmax_changed_fraction") if key in cells else None
        )
        sweep = row["receiver_changed_fraction"]
        result.append(
            {
                "model": key[0],
                "latent_step": key[1],
                "independent_run_id": cells[key].get("run_id", "")
                if key in cells
                else "",
                "independent_receiver_changed_fraction": original,
                "sweep_receiver_changed_fraction": sweep,
                "difference": sweep - original if original is not None else None,
                "matches": sweep == original if original is not None else None,
                "status": "checked" if original is not None else "not_available",
            }
        )
    return result
