"""Compare prompt and latent KV with common metric scales and a clear boundary."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIELDS = (
    ("norm", "Norm", None),
    ("adjacent_cosine", "Adjacent cosine", (-1, 1)),
    ("adjacent_l2", "Adjacent L2", None),
    ("nearest_prompt_cosine", "Nearest other prompt cosine", (-1, 1)),
)


def metric_limits(rows, component, field, fixed):
    if fixed is not None:
        return fixed
    values = [
        r[field]
        for r in rows
        if r["component"] == component and r.get(field) is not None
    ]
    if not values:
        return 0, 1
    low, high = min(values), max(values)
    return low, high if high > low else low + 1


def heatmap(axis, rows, layers, field, limits, coordinate):
    axis.set_facecolor("#cccccc")
    positions = sorted({r[coordinate] for r in rows})
    if not rows or not positions:
        axis.text(0.5, 0.5, "Unavailable", transform=axis.transAxes, ha="center")
        return
    lookup = {(r["layer"], r[coordinate]): r.get(field) for r in rows}
    matrix = np.array(
        [
            [
                np.nan if lookup.get((layer, p)) is None else lookup[layer, p]
                for p in positions
            ]
            for layer in layers
        ]
    )
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=plt.get_cmap("viridis").with_extremes(bad="#cccccc"),
        vmin=limits[0],
        vmax=limits[1],
        extent=(positions[0] - 0.5, positions[-1] + 0.5, len(layers) - 0.5, -0.5),
    )
    ticks = list(range(0, len(layers), max(1, len(layers) // 8)))
    axis.set_yticks(ticks, [layers[i] for i in ticks])
    axis.figure.colorbar(image, ax=axis)


def plot_kv(rows, summaries, sample, trajectory, output):
    layers = sorted({r["layer"] for r in rows})
    prompt_length = len({r["cache_position"] for r in rows if r["region"] == "prompt"})
    images = []
    fig, axes = plt.subplots(2, len(FIELDS), figsize=(20, 8), squeeze=False)
    for component, axes_row in zip(("K", "V"), axes, strict=True):
        selected = [r for r in rows if r["component"] == component]
        for axis, (field, label, fixed) in zip(axes_row, FIELDS, strict=True):
            heatmap(
                axis,
                selected,
                layers,
                field,
                metric_limits(rows, component, field, fixed),
                "cache_position",
            )
            if rows:
                axis.axvline(
                    prompt_length - 0.5, color="red", linestyle="--", linewidth=1.2
                )
            axis.set(
                title=f"{component}: {label}",
                xlabel="Cache position: prompt | latent",
                ylabel="Layer",
            )
    fig.suptitle(
        f"Sample {sample} | {trajectory} | full cache; red line=prompt/latent boundary"
    )
    fig.tight_layout()
    name = f"kv_{sample:06d}_{trajectory}.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    images.append(name)
    for component in ("K", "V"):
        fig, axes = plt.subplots(2, len(FIELDS), figsize=(20, 8), squeeze=False)
        for region, axes_row in zip(("prompt", "latent"), axes, strict=True):
            selected = [
                r for r in rows if r["component"] == component and r["region"] == region
            ]
            for axis, (field, label, fixed) in zip(axes_row, FIELDS, strict=True):
                heatmap(
                    axis,
                    selected,
                    layers,
                    field,
                    metric_limits(rows, component, field, fixed),
                    "region_position",
                )
                axis.set(
                    title=f"{region}: {label}",
                    xlabel="Prompt token index"
                    if region == "prompt"
                    else "Latent step",
                    ylabel="Layer",
                )
        fig.suptitle(
            f"Sample {sample} | {trajectory} | {component}: identical color scales within each column"
        )
        fig.tight_layout()
        name = f"kv_regions_{component}_{sample:06d}_{trajectory}.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        images.append(name)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    rank_fields = (
        ("effective_rank", "Full-region effective rank"),
        ("matched_effective_rank", "Equal-length window effective rank"),
        ("top1_energy_fraction", "Full-region leading energy fraction"),
    )
    for component, axes_row in zip(("K", "V"), axes, strict=True):
        selected = [r for r in summaries if r["component"] == component]
        for axis, (field, label) in zip(axes_row, rank_fields, strict=True):
            for region in ("prompt", "latent"):
                key = (
                    f"matched_{region}_effective_rank"
                    if field == "matched_effective_rank"
                    else f"{region}_{field}"
                )
                values = [r.get(key) for r in selected]
                axis.plot(
                    [r["layer"] for r in selected],
                    [np.nan if v is None else v for v in values],
                    marker=".",
                    label=region,
                )
            axis.set(xlabel="Layer", ylabel=label, title=component, facecolor="#eeeeee")
            axis.legend()
            axis.grid(alpha=0.25)
    width = next(
        (
            r["matched_window_positions"]
            for r in summaries
            if "matched_window_positions" in r
        ),
        None,
    )
    fig.suptitle(
        f"Sample {sample} | {trajectory} | uncentered rank\nMatched windows: prompt tail vs latent start, {width} positions each"
    )
    fig.tight_layout()
    name = f"kv_rank_{sample:06d}_{trajectory}.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    return *images, name
