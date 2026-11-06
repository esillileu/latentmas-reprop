"""Shared hidden/logit plots and their descriptive links to readout events."""

from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from latentmas_reprop.application.latent_trajectory_analysis import (
    join_readout_features,
)

from .latent_trajectory_report import write_csv

FEATURES = (
    "l2_norm",
    "distance_0",
    "adjacent_l2",
    "adjacent_cosine",
    "entropy",
    "js_divergence",
    "think_probability",
    "endthink_probability",
)


def write_state_report(readouts, latent, statuses, output):
    joined = join_readout_features(readouts, latent)
    write_csv(output / "latent_features.csv", latent)
    write_csv(output / "readout_latent_features.csv", joined)
    lookup = {
        (r["source_run_id"], r["sample_id"], r["condition"], r["step"]): r
        for r in joined
    }
    events = []
    for row in joined:
        if row.get("transition") not in ("0→1", "1→0"):
            continue
        before = lookup[
            row["source_run_id"], row["sample_id"], row["condition"], row["step"] - 1
        ]
        events.append(
            {**row, **{f"previous_{field}": before.get(field) for field in FEATURES}}
        )
    write_csv(output / "latent_events.csv", events)
    lines = [
        "# Shared latent states and readout events",
        "",
        "Each A/C or B/D pair shares one trajectory. Features are extracted once per trajectory.",
        "Raw logits precede closing-tag prefill; C/D actual continuation-start logits are unavailable.",
        "Step 0 hidden is prompt_hidden[-1]. Step 0 logits are not saved; logits use step 1 as their initial reference.",
        "Missing features remain empty/gray and curves never bridge missing steps.",
        "PCA is centered and fitted per trajectory; axes cannot be compared across independently fitted plots.",
        "Effective rank uses entropy of normalized nonzero singular values of centered states.",
        "No pooled significance tests are applied to correlated steps or duplicated shared states.",
        "Geometry/distribution changes and correctness associations do not establish reasoning or causality.",
        "",
        "Unavailable: actual injected-vector/embedding comparison (D4), conditional answer scoring (C5), step 0 and post-closing-prefill logits (weights required).",
        "",
        f"Correctness-transition events with joined features: {len(events)}. See latent_events.csv.",
        "",
        "| Sample | Trajectory | Artifact | Hidden available | Logits available | Effective rank |",
        "|---|---|---|---:|---:|---:|",
    ]
    images = []
    for status in statuses:
        selected = [
            r
            for r in latent
            if r["sample_id"] == status["sample_id"]
            and r["trajectory_type"] == status["trajectory_type"]
        ]
        hidden = Counter(r["hidden_status"] for r in selected)
        logits = Counter(r["logits_status"] for r in selected)
        rank = next(
            (r["effective_rank"] for r in selected if "effective_rank" in r), None
        )
        lines.append(
            f"| {status['sample_id']} | {status['trajectory_type']} | {status['status']} | {hidden['complete']}/{len(selected)} | {logits['complete']}/{len(selected)} | {rank if rank is not None else '—'} |"
        )
        if selected:
            names = plot_shared_states(selected, joined, output)
            images.extend(names)
    for name in images:
        lines.extend(["", f"![Shared trajectory]({name})", ""])
    (output / "latent_summary.md").write_text("\n".join(lines) + "\n")
    with (output / "summary.md").open("a") as stream:
        stream.write(
            "\n\n[Shared hidden/logit features and readout events](latent_summary.md)\n"
        )


def plot_shared_states(rows, joined, output):
    sample, trajectory = rows[0]["sample_id"], rows[0]["trajectory_type"]
    conditions = "ac" if trajectory == "without_think" else "bd"
    steps = [row["step"] for row in rows]
    readouts = {
        (r["condition"], r["step"]): r for r in joined if r["sample_id"] == sample
    }
    fig, axes = plt.subplots(7, 1, figsize=(13, 16), sharex=True)
    colors = ListedColormap(["#bdbdbd", "#d95f59", "#54a86b"])
    matrix = [
        [
            int(readouts[c, k]["evaluator_correct"])
            if (c, k) in readouts and readouts[c, k]["status"] == "complete"
            else -1
            for k in steps
        ]
        for c in conditions
    ]
    axes[0].imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=colors,
        norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3),
        extent=(steps[0] - 0.5, steps[-1] + 0.5, 1.5, -0.5),
    )
    axes[0].set_yticks([0, 1], list(conditions.upper()))
    axes[0].set_title(
        "Readout correctness: green=correct, red=incorrect, gray=unavailable"
    )
    plots = (
        ("l2_norm", "Hidden norm"),
        ("distance_0", "Distance from step 0"),
        ("entropy", "Raw-logit entropy (nats)"),
        ("js_divergence", "Adjacent JS (nats)"),
        ("think_probability", "Raw P(<think>)"),
        ("endthink_probability", "Raw P(</think>)"),
    )
    for axis, (field, label) in zip(axes[1:], plots, strict=True):
        values = [r.get(field) for r in rows]
        axis.set_facecolor("#eeeeee")
        axis.plot(
            steps, [np.nan if v is None else v for v in values], marker=".", linewidth=1
        )
        if all(v is None for v in values):
            axis.text(0.5, 0.5, "Unavailable", transform=axis.transAxes, ha="center")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Latent step")
    fig.suptitle(
        f"{rows[0]['benchmark']} sample {sample} | {trajectory} | alignment={rows[0]['alignment']}"
    )
    fig.tight_layout()
    name = f"latent_{sample:06d}_{trajectory}.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    geometry = [r for r in rows if r.get("pca_1") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if geometry:
        x, y = [r["pca_1"] for r in geometry], [r["pca_2"] for r in geometry]
        axes[0].plot(x, y, alpha=0.4)
        points = axes[0].scatter(x, y, c=[r["step"] for r in geometry], cmap="viridis")
        fig.colorbar(points, ax=axes[0], label="Latent step")
        axes[0].annotate(str(geometry[0]["step"]), (x[0], y[0]))
    else:
        axes[0].text(
            0.5, 0.5, "PCA unavailable", transform=axes[0].transAxes, ha="center"
        )
    pairs = [
        r
        for r in rows
        if r.get("adjacent_l2") is not None and r.get("js_divergence") is not None
    ]
    if pairs:
        points = axes[1].scatter(
            [r["adjacent_l2"] for r in pairs],
            [r["js_divergence"] for r in pairs],
            c=[r["step"] for r in pairs],
            cmap="viridis",
        )
        fig.colorbar(points, ax=axes[1], label="Latent step")
    else:
        axes[1].text(
            0.5,
            0.5,
            "Paired features unavailable",
            transform=axes[1].transAxes,
            ha="center",
        )
    axes[0].set(xlabel="Centered PC1", ylabel="Centered PC2")
    axes[1].set(xlabel="Adjacent hidden L2", ylabel="Adjacent logit JS (nats)")
    for axis in axes:
        axis.set_facecolor("#eeeeee")
        axis.grid(alpha=0.2)
    fig.tight_layout()
    geometry_name = f"geometry_{sample:06d}_{trajectory}.png"
    fig.savefig(output / geometry_name, dpi=150)
    plt.close(fig)
    return name, geometry_name
