"""Fit and plot all saved items on common axes, separately for each model."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from latentmas_reprop.application.latent_global_pca import analyze_global_pca
from latentmas_reprop.infrastructure.storage import save_json_atomic, save_tensor

from .latent_common_pca_report import COLORS
from .latent_trajectory_report import write_csv


def write_global_report(result, output):
    output.mkdir(parents=True, exist_ok=True)
    coordinates = result.get("coordinates", [])
    serial = {
        key: value
        for key, value in result.items()
        if key not in ("basis", "coordinates", "groups")
    }
    serial["groups"] = [
        {key: value for key, value in group.items() if key != "hidden"}
        for group in result["groups"]
    ]
    if "basis" in result:
        save_tensor(output / "global_pca.pt", result["basis"], overwrite=True)
        serial["explained_variance_ratio"] = result["basis"][
            "explained_variance_ratio"
        ].tolist()
    else:
        (output / "global_pca.pt").unlink(missing_ok=True)
    save_json_atomic(output / "global_pca_status.json", serial)
    write_csv(output / "global_pca_coordinates.csv", coordinates)
    plot_global_pca(result, output)
    items = {(r["benchmark"], r["sample_id"]) for r in coordinates}
    sources = {(r["benchmark"], r["source_run_id"]) for r in coordinates}
    lines = [
        "# Global latent PCA",
        "",
        f"Model: {result['model']}; status: {result['status']}.",
        "",
        "All saved available states are trusted as-is. No freshness checks, current-run filtering, warnings about age, or automatic refresh are applied.",
        "One mean and basis covers all included benchmarks/items/conditions of this model. Benchmark panels use these same axes and limits.",
        "Fit weights balance benchmarks, then items, then saved shared trajectories, then states. A/C and B/D are not duplicated.",
        "Different model spaces are fitted separately. No individual item/trajectory recentering is applied.",
        "",
        f"Included items: {len(items)}; source runs: {len(sources)}; states: {len(coordinates)}.",
        "",
    ]
    if "explained_variance_ratio" in serial:
        ratio = serial["explained_variance_ratio"]
        lines.append(
            f"Explained variance: PC1 {ratio[0]:.1%}, PC2 {ratio[1]:.1%}, total {sum(ratio):.1%}."
        )
    if "error" in result:
        lines.append(f"Unavailable: {result['error']}")
    lines.extend(
        [
            "",
            "![Global PCA](global_pca.png)",
            "",
            "| Benchmark | Item | Run | Alignment | Trajectory | States | Status |",
            "|---|---|---|---|---|---:|---|",
        ]
    )
    for group in result["groups"]:
        lines.append(
            f"| {group['benchmark']} | {group['sample_id']} | {group['source_run_id']} | {'ON' if group['alignment'] else 'OFF'} | {group['trajectory_type']} | {len(group['steps'])} | {group['status']} |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def plot_global_pca(result, output):
    tasks = sorted({t["benchmark"] for t in result["groups"]})
    fig, axes = plt.subplots(
        1,
        1 + len(tasks),
        figsize=(7 * (1 + len(tasks)), 7),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    coordinates = result.get("coordinates", [])
    groups = {}
    for row in coordinates:
        key = (
            row["benchmark"],
            row["sample_id"],
            row["source_run_id"],
            row["alignment"],
            row["trajectory_type"],
        )
        groups.setdefault(key, []).append(row)
    ratio = (
        result["basis"]["explained_variance_ratio"].tolist()
        if "basis" in result
        else None
    )
    for axis, task in zip(axes[0], [None, *tasks], strict=True):
        for key, rows in groups.items():
            benchmark, sample, _, alignment, trajectory = key
            if task is not None and task != benchmark:
                continue
            x, y = [r["common_pca_1"] for r in rows], [r["common_pca_2"] for r in rows]
            color = COLORS[alignment, trajectory]
            axis.plot(
                x,
                y,
                color=color,
                linestyle="--" if trajectory == "with_think" else "-",
                linewidth=0.8,
                alpha=0.5,
            )
            axis.scatter(x, y, color=color, s=5, alpha=0.25)
            axis.scatter(x[0], y[0], color=color, marker="s", s=25)
            axis.scatter(x[-1], y[-1], color=color, marker=">", s=35)
            axis.annotate(
                f"{benchmark}:{sample}", (x[-1], y[-1]), color=color, fontsize=7
            )
        if not coordinates:
            axis.text(
                0.5,
                0.5,
                "No saved hidden states",
                transform=axis.transAxes,
                ha="center",
            )
        axis.set_title("All benchmarks" if task is None else task)
        axis.set_xlabel(
            f"Global PC1 ({ratio[0]:.1%})" if ratio else "Global PC1 unavailable"
        )
        axis.set_ylabel(
            f"Global PC2 ({ratio[1]:.1%})" if ratio else "Global PC2 unavailable"
        )
        axis.set_aspect("equal", adjustable="box")
        axis.set_facecolor("#eeeeee")
        axis.grid(alpha=0.2)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle="--" if trajectory == "with_think" else "-",
            label=f"{'ON' if alignment else 'OFF'} / {trajectory}",
        )
        for (alignment, trajectory), color in COLORS.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4)
    variance = f"{sum(ratio):.1%}" if ratio else "unavailable"
    fig.suptitle(
        f"{result['model']} | Global PCA | explained total {variance}\nOne basis and identical limits across all panels; square=start, triangle=end"
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    fig.savefig(output / "global_pca.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("storage/experiments/latent_trajectory"),
    )
    parser.add_argument(
        "--model", help="Exact model name; by default fit each model separately"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/latent_trajectory/global_pca"),
    )
    args = parser.parse_args(argv)
    results = analyze_global_pca(args.directory.resolve(), args.model)
    for result in results:
        output = args.output_dir / result["model"]
        write_global_report(result, output)
        print(f"Global PCA: {output.resolve() / 'summary.md'}")
    if not results:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.md").write_text(
            "# Global latent PCA\n\nNo saved items found.\n"
        )
        print(f"Global PCA: {args.output_dir.resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
