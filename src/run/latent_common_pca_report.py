"""Condition-comparison PCA plots with one common basis per model and item."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from latentmas_reprop.infrastructure.storage import save_json_atomic, save_tensor

from .latent_trajectory_report import write_csv

COLORS = {
    (False, "without_think"): "#2471a3",
    (False, "with_think"): "#17a589",
    (True, "without_think"): "#cb4335",
    (True, "with_think"): "#d68910",
}


def write_common_pca_report(results, output):
    coordinates, statuses, images = [], [], []
    lines = [
        "# Per-item common PCA",
        "",
        "Primary condition comparison: ON/OFF x with/without-think share one mean and one PCA basis within each item.",
        "The same procedure applies to GSM8K and MedQA; different items/benchmarks are not pooled into one basis.",
        "Each available shared trajectory has equal total fit weight; states within it have equal weight. A/C and B/D are not duplicated.",
        "Individual PCA plots remain supplementary within-trajectory views with different axes.",
        "Missing/incompatible conditions are gray in the legend and excluded from fitting. Step 0 is included when available.",
        "A basis is refitted when its available source set changes. Compare coordinates only within the same saved basis, not across items or report generations.",
        "",
        "| Sample | Fit status | Trajectories | PC1 | PC2 | Total explained variance |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        sample = result["sample_id"]
        serial = {
            key: value
            for key, value in result.items()
            if key not in ("basis", "coordinates", "groups")
        }
        serial["groups"] = [
            {key: value for key, value in group.items() if key != "hidden"}
            for group in result["groups"]
        ]
        available = sum("hidden" in group for group in result["groups"])
        ratio = [0.0, 0.0]
        if "basis" in result:
            relative = f"common_pca_{sample:06d}.pt"
            save_tensor(output / relative, result["basis"], overwrite=True)
            ratio = result["basis"]["explained_variance_ratio"].tolist()
            serial.update(basis_path=relative, explained_variance_ratio=ratio)
            coordinates.extend(
                {
                    "benchmark": result["benchmark"],
                    "model": result["model"],
                    "basis_path": relative,
                    **row,
                }
                for row in result["coordinates"]
            )
        else:
            (output / f"common_pca_{sample:06d}.pt").unlink(missing_ok=True)
        statuses.append(serial)
        variance = (
            [f"{v:.1%}" for v in (*ratio, sum(ratio))]
            if "basis" in result
            else ["—"] * 3
        )
        lines.append(
            f"| {sample} | {result['status']} | {available}/4 | {' | '.join(variance)} |"
        )
        images.append(plot_common_pca(result, output, ratio))
    for image in images:
        lines.extend(["", f"![Common PCA]({image})", ""])
    for status in statuses:
        lines.extend([f"## Sample {status['sample_id']} availability", ""])
        lines.extend(f"- {note}" for note in status["notes"])
        if "error" in status:
            lines.append(f"- PCA unavailable: {status['error']}")
        for group in status["groups"]:
            label = (
                f"{'ON' if group['alignment'] else 'OFF'} / {group['trajectory_type']}"
            )
            lines.append(
                f"- {label}: {group['status']}; {len(group['steps'])} states. {'; '.join(group['errors'])}"
            )
    write_csv(output / "common_pca_coordinates.csv", coordinates)
    save_json_atomic(output / "common_pca_status.json", statuses)
    (output / "common_pca_summary.md").write_text("\n".join(lines) + "\n")
    for name in ("summary.md", "latent_summary.md"):
        with (output / name).open("a") as stream:
            stream.write(
                "\n\n[Primary condition comparison: common PCA](common_pca_summary.md)\n"
            )


def plot_common_pca(result, output, ratio):
    fig, axis = plt.subplots(figsize=(11, 8))
    handles = []
    for group in result["groups"]:
        label = f"{'ON' if group['alignment'] else 'OFF'} / {group['trajectory_type']}"
        rows = [
            r
            for r in result.get("coordinates", [])
            if r["alignment"] == group["alignment"]
            and r["trajectory_type"] == group["trajectory_type"]
        ]
        color = (
            COLORS[group["alignment"], group["trajectory_type"]] if rows else "#aaaaaa"
        )
        style = "--" if group["trajectory_type"] == "with_think" else "-"
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linestyle=style,
                label=f"{label} ({len(rows)} states)"
                if rows
                else f"{label}: {group['status']}",
            )
        )
        if not rows:
            continue
        x, y = [r["common_pca_1"] for r in rows], [r["common_pca_2"] for r in rows]
        axis.plot(x, y, color=color, linestyle=style, alpha=0.8, linewidth=1.2)
        axis.scatter(x, y, color=color, s=12, alpha=0.5)
        axis.scatter(x[0], y[0], color=color, marker="s", s=55)
        axis.scatter(x[-1], y[-1], color=color, marker=">", s=65)
        for row in rows:
            if row["step"] % 20 == 0:
                axis.annotate(
                    str(row["step"]),
                    (row["common_pca_1"], row["common_pca_2"]),
                    color=color,
                    fontsize=8,
                    xytext=(4, 4),
                    textcoords="offset points",
                )
    if "basis" not in result:
        axis.text(
            0.5, 0.5, "Common PCA unavailable", transform=axis.transAxes, ha="center"
        )
    variance_labels = (
        [f"{value:.1%}" for value in ratio]
        if "basis" in result
        else ["unavailable"] * 2
    )
    axis.set(
        xlabel=f"Common PC1 ({variance_labels[0]})",
        ylabel=f"Common PC2 ({variance_labels[1]})",
    )
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_facecolor("#eeeeee")
    axis.grid(alpha=0.25)
    axis.legend(handles=handles, fontsize=9, loc="best")
    total = f"{sum(ratio):.1%}" if "basis" in result else "unavailable"
    axis.set_title(
        f"{result['benchmark']} | {result['model']} | sample {result['sample_id']}\nCommon PCA: {result['status']} | explained total {total}\nSquare=start; triangle=end; numbers=latent steps"
    )
    fig.tight_layout()
    name = f"common_pca_{result['sample_id']:06d}.png"
    fig.savefig(output / name, dpi=180)
    plt.close(fig)
    return name
