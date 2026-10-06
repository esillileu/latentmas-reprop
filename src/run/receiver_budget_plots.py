"""Draw finite solver budget curves using saved paired analysis results."""

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .receiver_budget_data import percent, read_csv, unique_grid

MODELS = ("4B", "8B", "14B")
STEPS = (10, 20)
BUDGETS = (64, 128, 256, 512, 1024)
CONDITIONS = (
    ("matched", "Matched latent", "#0072B2", "o", "-"),
    ("mismatched", "Mismatched latent", "#D55E00", "s", "--"),
    ("no_handoff", "No latent", "#666666", "^", ":"),
)


def read_budget_curves(compute, sample_count=None):
    """Require complete finite grids and consistent saved effect estimates."""
    results = {}
    for model in MODELS:
        directory = compute / f"Qwen_Qwen3-{model}"
        source = json.loads((directory / "source.json").read_text())
        if source["model"] != f"Qwen/Qwen3-{model}":
            raise ValueError(f"Unexpected model provenance in {directory}")
        path = directory / "budget_curves.csv"
        curves = unique_grid(
            [r for r in read_csv(path) if r["receiver_budget"] != "free"],
            lambda r: (
                int(r["upstream_steps"]),
                r["handoff_condition"],
                int(r["receiver_budget"]),
            ),
            [(u, c[0], b) for u in STEPS for c in CONDITIONS for b in BUDGETS],
            path,
        )
        metrics_path = directory / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        effects = unique_grid(
            [
                r
                for r in metrics["comparisons"]
                if r["comparison"] == "pairing_gain" and r["receiver_budget"] != "free"
            ],
            lambda r: (int(r["upstream_steps"]), int(r["receiver_budget"])),
            [(u, b) for u in STEPS for b in BUDGETS],
            metrics_path,
        )
        counts = {int(r["sample_count"]) for r in curves.values()}
        if counts != {metrics["sample_count"]} or (
            sample_count is not None and counts != {sample_count}
        ):
            raise ValueError(f"Inconsistent sample_count in {directory}")
        for row in curves.values():
            percent(row, "accuracy")
        for b in BUDGETS:
            if (
                curves[10, "no_handoff", b]["accuracy"]
                != curves[20, "no_handoff", b]["accuracy"]
            ):
                raise ValueError(f"No latent baseline differs between panels in {path}")
        for (u, b), row in effects.items():
            value, low, high = (
                float(row[k]) for k in ("difference", "ci_low", "ci_high")
            )
            if (
                not all(math.isfinite(v) for v in (value, low, high))
                or not -1 <= low <= value <= high <= 1
            ):
                raise ValueError(f"Invalid paired CI in {metrics_path}")
            expected = float(curves[u, "matched", b]["accuracy"]) - float(
                curves[u, "mismatched", b]["accuracy"]
            )
            if not math.isclose(value, expected, abs_tol=1e-12):
                raise ValueError(f"Saved effect disagrees with accuracies in {path}")
        results[model] = (curves, effects)
    return results


def panels(model, title, ylabel, ylim):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.19, top=0.70, wspace=0.12)
    fig.suptitle(f"Qwen3-{model}: {title}", fontsize=28, y=0.97)
    for ax, steps in zip(axes, STEPS, strict=True):
        ax.set_xscale("log", base=2)
        ax.set_xticks(BUDGETS, [str(b) for b in BUDGETS])
        ax.set(
            title=f"Upstream latent steps = {steps}",
            xlabel="Solver generation budget",
            ylim=ylim,
        )
        ax.grid(color="#e5e5e5", linewidth=0.8)
        ax.set_facecolor("white")
    axes[0].set_ylabel(ylabel)
    fig.set_facecolor("white")
    return fig, axes


def save(fig, output, name):
    for extension in ("svg", "png"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        print(path)
    plt.close(fig)


def plot_budget(results, output):
    bounds = [
        100 * float(row[k])
        for _, effects in results.values()
        for row in effects.values()
        for k in ("ci_low", "ci_high")
    ]
    low, high = min(0, min(bounds)), max(0, max(bounds))
    padding = max(2, (high - low) * 0.08)
    effect_limits = (low - padding, high + padding)
    for model, (curves, effects) in results.items():
        fig, axes = panels(
            model, "Accuracy across Solver Budgets", "Accuracy (%)", (0, 100)
        )
        for ax, steps in zip(axes, STEPS, strict=True):
            for condition, label, color, marker, linestyle in CONDITIONS:
                baseline_steps = 10 if condition == "no_handoff" else steps
                ax.plot(
                    BUDGETS,
                    [
                        percent(curves[baseline_steps, condition, b], "accuracy")
                        for b in BUDGETS
                    ],
                    label=label,
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    markersize=9,
                    linewidth=3,
                )
            ax.set_yticks(range(0, 101, 20))
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.90),
            ncol=3,
            frameon=False,
            fontsize=22,
        )
        save(fig, output, f"receiver_budget_{model.lower()}")
        fig, axes = panels(
            model,
            "Content Effect across Solver Budgets",
            "Content effect (pp)",
            effect_limits,
        )
        fig.text(
            0.5,
            0.88,
            "Matched \N{MINUS SIGN} Mismatched accuracy",
            ha="center",
            fontsize=20,
            color="#555555",
        )
        for ax, steps in zip(axes, STEPS, strict=True):
            rows = [effects[steps, b] for b in BUDGETS]
            values = [100 * float(r["difference"]) for r in rows]
            ax.axhline(0, color="#888888", linewidth=1.5)
            ax.errorbar(
                BUDGETS,
                values,
                yerr=[
                    [
                        v - 100 * float(r["ci_low"])
                        for v, r in zip(values, rows, strict=True)
                    ],
                    [
                        100 * float(r["ci_high"]) - v
                        for v, r in zip(values, rows, strict=True)
                    ],
                ],
                fmt="o",
                color=CONDITIONS[0][2],
                markersize=9,
                capsize=6,
                elinewidth=2,
            )
        fig.text(
            0.5,
            0.045,
            "Paired bootstrap 95% CI; no multiplicity correction reported.",
            ha="center",
            fontsize=15,
            color="#555555",
        )
        save(fig, output, f"receiver_content_effect_{model.lower()}")
