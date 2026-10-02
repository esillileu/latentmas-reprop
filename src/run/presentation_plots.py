"""Draw presentation figures from saved analysis and supplied data; no inference."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .secret_digit_plots import create_digit_aspects_figure

BUDGETS = (64, 128, 256, 512, 1024)
CONDITIONS = (
    ("matched", "Matched", "#0072B2", "o", "-"),
    ("mismatched", "Mismatched", "#D55E00", "s", "--"),
    ("no_handoff", "No Handoff", "#666666", "^", ":"),
)
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
    curves = {}
    for model in ("4B", "14B"):
        path = compute / f"Qwen_Qwen3-{model}" / "budget_curves.csv"
        source = json.loads(path.with_name("source.json").read_text())
        if source["model"] != f"Qwen/Qwen3-{model}":
            raise ValueError(f"Unexpected model provenance in {path.parent}")
        curves[model] = unique_grid(
            [
                r
                for r in read_csv(path)
                if r["upstream_steps"] == "20"
                and r["receiver_budget"] in {str(b) for b in BUDGETS}
                and r["handoff_condition"] in {c[0] for c in CONDITIONS}
            ],
            lambda r: (r["handoff_condition"], int(r["receiver_budget"])),
            [(c[0], b) for c in CONDITIONS for b in BUDGETS],
            path,
        )
    return probe, receiver, curves


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


def plot_budget(curves, output):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.80, wspace=0.16)
    for ax, model in zip(axes, ("4B", "14B"), strict=True):
        for condition, label, color, marker, linestyle in CONDITIONS:
            rows = [curves[model][condition, budget] for budget in BUDGETS]
            values = [percent(r, "accuracy") for r in rows]
            ax.plot(
                BUDGETS,
                values,
                color=color,
                marker=marker,
                markersize=11,
                linewidth=4,
                linestyle=linestyle,
                label=label,
            )
        ax.axhline(50, color="#999999", linestyle="--", linewidth=1.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set_yticks(range(0, 101, 20))
        ax.set(title=model, xlabel="Receiver reasoning budget", ylim=(0, 100))
        ax.tick_params(labelleft=True)
        ax.grid(axis="y", alpha=0.18)
    axes[0].set_ylabel("GSM8K accuracy (%)")
    fig.suptitle("Receiver budget · 20 upstream steps", fontsize=28, y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=False,
        prop={"size": 22, "weight": "bold"},
        handlelength=3,
    )
    save(fig, output, "receiver_budget_4b_14b")


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
    args = parser.parse_args(argv)
    probe, receiver, curves = prepare_inputs(args.acquisition_dir, args.compute_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        plot_probe(probe, args.output_dir)
        plot_receiver(receiver, args.output_dir)
        plot_budget(curves, args.output_dir)
        save(create_digit_aspects_figure(), args.output_dir, "secret_digit_aspects")


if __name__ == "__main__":
    main()
