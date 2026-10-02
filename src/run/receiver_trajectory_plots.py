"""Read-only presentation scatter of all 60 probe/receiver trajectory cells."""

import argparse
import csv
import math
from pathlib import Path

from .presentation_plots import STYLE, plt, save


def plot_trajectory(path, output):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    models = ("0.6B", "4B", "8B")
    expected = {(f"Qwen/Qwen3-{m}", k) for m in models for k in range(1, 21)}
    if (
        len(rows) != 60
        or {(r["model"], int(r["latent_step"])) for r in rows} != expected
    ):
        raise ValueError("The presentation requires exactly 60 unique model/step cells")
    if any(r["sample_count"] != "100" for r in rows):
        raise ValueError("Smoke results cannot be used in the 60-point presentation")
    fig, ax = plt.subplots(figsize=(16, 9), layout="constrained")
    all_x = []
    for model, color in zip(models, ("#7B3294", "#0072B2", "#D55E00"), strict=True):
        cells = sorted(
            (r for r in rows if r["model"] == f"Qwen/Qwen3-{model}"),
            key=lambda r: int(r["latent_step"]),
        )
        xs = [float(r["probe_effect_pp"]) for r in cells]
        ys = [100 * float(r["receiver_changed_fraction"]) for r in cells]
        if any(not math.isfinite(x) for x in xs) or any(not 0 <= y <= 100 for y in ys):
            raise ValueError("Invalid scatter coordinates")
        all_x.extend(xs)
        ax.plot(xs, ys, color=color, linewidth=1.1, alpha=0.5)
        for row, x, y in zip(cells, xs, ys, strict=True):
            ax.plot(
                x,
                y,
                marker="o",
                markersize=10,
                markeredgewidth=2,
                markeredgecolor=color,
                markerfacecolor=color if float(row["probe_fwer_p"]) < 0.05 else "white",
                linestyle="none",
                clip_on=False,
            )
        ax.plot(
            [],
            [],
            marker="o",
            color=color,
            linestyle="none",
            label=model,
            markersize=10,
        )
    ax.set(
        xlabel="Digit signal detectable in handoff latent\n(pp above null)",
        ylabel="Receiver predictions changed by handoff (%)",
        ylim=(0, 100),
        xlim=(math.floor(min(all_x)) - 0.5, math.ceil(max(all_x)) + 0.5),
    )
    ax.set_title(
        "Sender digit signal and receiver response across latent steps", pad=65
    )
    ax.set_yticks(range(0, 101, 20))
    ax.grid(False)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=3,
        prop={"size": 22, "weight": "bold"},
    )
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "secret_digit_trajectory")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/receiver_trajectory/trajectory.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/presentation")
    )
    args = parser.parse_args(argv)
    with plt.rc_context(STYLE):
        plot_trajectory(args.input, args.output_dir)


if __name__ == "__main__":
    main()
