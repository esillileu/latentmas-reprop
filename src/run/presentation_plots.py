"""Draw presentation figures from saved analysis and supplied data; no inference."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .receiver_budget_data import percent, read_csv, unique_grid
from .receiver_budget_plots import plot_budget, read_budget_curves
from .secret_digit_plots import create_digit_aspects_figure

STYLE = {
    "font.size": 20,
    "axes.titlesize": 26,
    "axes.labelsize": 22,
    "legend.fontsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "lines.linewidth": 3,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
}


def digit_rows(path):
    return [
        row
        for row in read_csv(path)
        if row["model"] == "Qwen/Qwen3-8B" and row["latent_steps"] == "20"
    ]


def prepare_inputs(acquisition, compute):
    representations = ("hidden_pre_realign", "latent_post_realign")
    probe_path = acquisition / "sender_probe_cells.csv"
    probe = unique_grid(
        [r for r in digit_rows(probe_path) if r["representation"] in representations],
        lambda r: (r["representation"], int(r["step"])),
        [(rep, step) for rep in representations for step in range(1, 21)],
        probe_path,
    )
    receiver_path = acquisition / "receiver_conditions.csv"
    receiver_conditions = ("drop", "latent_only/own", "latent_only/cross")
    receiver = unique_grid(
        [r for r in digit_rows(receiver_path) if r["condition"] in receiver_conditions],
        lambda r: r["condition"],
        receiver_conditions,
        receiver_path,
    )
    return probe, receiver, read_budget_curves(compute)


def save(fig, output, name):
    for extension in ("png", "pdf"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def plot_probe(probe, output):
    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
    steps = list(range(1, 21))
    for rep, label, color, marker, linestyle in (
        ("hidden_pre_realign", "Pre-realign", "#0072B2", "o", "-"),
        ("latent_post_realign", "Post-realign", "#D55E00", "s", "--"),
    ):
        rows = [probe[rep, step] for step in steps]
        values = [percent(r, "probe_accuracy") for r in rows]
        ax.plot(steps, values, color=color, linestyle=linestyle, label=label)
        for step, value, row in zip(steps, values, rows, strict=True):
            if row["significance"].lower() not in {"true", "false"}:
                raise ValueError("Probe significance must be True or False")
            ax.plot(
                step,
                value,
                marker=marker,
                markersize=8,
                markeredgecolor=color,
                markeredgewidth=1.8,
                markerfacecolor=color
                if row["significance"].lower() == "true"
                else "white",
            )
    ax.axhline(10, color="#777777", linestyle=":", linewidth=2, label="Chance (10%)")
    ax.set(
        xlabel="Latent step",
        ylabel="Digit probe accuracy (%)",
        title="Secret-digit probe · Qwen3-8B",
        xlim=(0.5, 20.5),
        ylim=(0, 30),
    )
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks(range(0, 31, 5))
    ax.grid(axis="y", alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([], [], color="#333333", marker="o", linestyle="none")]
    labels += ["Filled: significant; hollow: non-significant"]
    ax.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    save(fig, output, "secret_digit_probe")


def plot_receiver(receiver, output):
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    labels = ["No Handoff", "Matched", "Mismatched"]
    values = [
        percent(receiver[c], "accuracy")
        for c in ("drop", "latent_only/own", "latent_only/cross")
    ]
    ax.bar(labels, values, color=["#666666", "#0072B2", "#D55E00"], width=0.58)
    ax.axhline(10, color="#444444", linestyle=":", linewidth=2.5, label="Chance (10%)")
    ax.set(
        ylabel="Receiver digit accuracy (%)",
        ylim=(0, 25),
        title="Secret-digit receiver · Qwen3-8B",
    )
    ax.set_yticks(range(0, 26, 5))
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, loc="upper right")
    save(fig, output, "secret_digit_receiver")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acquisition-dir", type=Path, default=Path("artifacts/receiver_acquisition")
    )
    parser.add_argument(
        "--compute-dir", type=Path, default=Path("artifacts/receiver_compute_preflight")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/presentation")
    )
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument("--sample-count", type=int, default=None)
    args = parser.parse_args(argv)
    if args.budget_only:
        curves = read_budget_curves(args.compute_dir, args.sample_count)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with plt.rc_context(STYLE):
            plot_budget(curves, args.output_dir)
        return
    probe, receiver, curves = prepare_inputs(args.acquisition_dir, args.compute_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        plot_probe(probe, args.output_dir)
        plot_receiver(receiver, args.output_dir)
        plot_budget(curves, args.output_dir)
        save(create_digit_aspects_figure(), args.output_dir, "secret_digit_aspects")


if __name__ == "__main__":
    main()
