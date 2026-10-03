"""Validate and plot saved sender/receiver metrics only; never run inference."""

import argparse
import csv
import json
import math
import os
import shutil
import tempfile
from pathlib import Path

from .presentation_plots import STYLE, plt, save

MODELS = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B")
COLORS = ("#7B3294", "#0072B2", "#D55E00", "#009E73")
COLUMNS = (
    "model",
    "latent_step",
    "probe_accuracy",
    "probe_null_mean",
    "probe_effect_pp",
    "probe_fwer_p",
    "probe_significant",
    "receiver_changed_fraction",
    "receiver_changed_pct",
)


def write_csv(path, rows, columns=None):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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


def plot_metrics(rows, output, annotate=False):
    models = models_in(rows)
    colors = COLORS[: len(models)]
    effects = [row["probe_effect_pp"] for row in rows]
    signal_start = 2 * math.floor(min(0, min(effects)) / 2)
    signal_end = 2 * math.ceil(max(0, max(effects)) / 2)
    signal_ticks = range(signal_start, max(signal_start + 2, signal_end) + 1, 2)
    signal_limits = (signal_ticks[0] - 0.5, signal_ticks[-1] + 0.5)
    receiver_limits = (-3, 103)
    receiver_ticks = range(0, 101, 20)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")
        for model, color in zip(models, colors, strict=True):
            cells = [row for row in rows if row["model"] == model]
            xs = [row["probe_effect_pp"] for row in cells]
            ys = [row["receiver_changed_pct"] for row in cells]
            ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.5)
            for significant in (True, False):
                subset = [
                    row for row in cells if row["probe_significant"] == significant
                ]
                ax.scatter(
                    [row["probe_effect_pp"] for row in subset],
                    [row["receiver_changed_pct"] for row in subset],
                    s=110,
                    edgecolors=color,
                    facecolors=color if significant else "none",
                    linewidths=1.8,
                    zorder=3,
                )
            ax.plot([], [], "o", color=color, label=model.split("-")[-1])
            if annotate:
                for row in cells:
                    if row["latent_step"] in (1, 4, 20):
                        ax.annotate(
                            str(row["latent_step"]),
                            (row["probe_effect_pp"], row["receiver_changed_pct"]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=11,
                        )
        ax.set(
            xlabel="Digit signal in handoff latent (pp above null)",
            ylabel="Receiver predictions changed by handoff (%)",
            xlim=signal_limits,
            xticks=signal_ticks,
            ylim=receiver_limits,
            yticks=receiver_ticks,
        )
        ax.xaxis.label.set_fontsize(18)
        ax.yaxis.label.set_fontsize(16)
        ax.grid(False)
        ax.legend(frameon=False, ncol=len(models), fontsize=14)
        save(fig, output, "stepwise_communication_scatter")
    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(
            2,
            len(models),
            figsize=(4.7 * len(models), 7),
            layout="constrained",
            sharex=True,
            sharey="row",
        )
        for column, (model, color) in enumerate(zip(models, colors, strict=True)):
            cells = [row for row in rows if row["model"] == model]
            for index, field in enumerate(("probe_effect_pp", "receiver_changed_pct")):
                ax = axes[index, column]
                ax.plot(
                    [row["latent_step"] for row in cells],
                    [row[field] for row in cells],
                    "o-",
                    color=color,
                    linewidth=1,
                )
                ax.set(
                    title=model,
                    xlabel="latent_step",
                    ylabel=field,
                    xlim=(0.5, 20.5),
                    xticks=(1, 4, 10, 15, 20),
                    ylim=signal_limits if index == 0 else receiver_limits,
                    yticks=signal_ticks if index == 0 else receiver_ticks,
                )
                ax.grid(alpha=0.25)
        save(fig, output, "stepwise_communication_diagnostics")


def analyze(probes, sweep, independent, output, annotate=False):
    rows = communication_metrics(probes, sweep)
    parity = receiver_parity(rows, independent)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "stepwise_communication_metrics.csv", rows, COLUMNS)
    write_csv(output / "stepwise_receiver_parity.csv", parity)
    if any(row["matches"] is False for row in parity):
        raise ValueError(
            f"Receiver metric parity failed; inspect {output / 'stepwise_receiver_parity.csv'}"
        )
    plot_metrics(rows, output, annotate)
    return rows


def main(argv=None):
    from dotenv import load_dotenv
    from mlflow.tracking import MlflowClient

    from .stepwise_communication_sources import load_inputs

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--probe-run-ids", nargs="+", required=True)
    parser.add_argument("--sweep-run-ids", nargs="+", required=True)
    parser.add_argument("--independent-run-ids", nargs="+", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional local export of derived artifacts; never an input",
    )
    parser.add_argument("--annotate-steps", action="store_true")
    args = parser.parse_args(argv)
    uri = args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri or not uri.strip():
        parser.error("Set MLFLOW_TRACKING_URI or --tracking-uri; no local fallback")
    client = MlflowClient(tracking_uri=uri)
    probes, sweep, independent, provenance = load_inputs(
        client, args.probe_run_ids, args.sweep_run_ids, args.independent_run_ids
    )
    experiment = client.get_experiment_by_name("latentmas_receiver_trajectory")
    run = client.create_run(
        experiment.experiment_id,
        tags={
            "mlflow.runName": "stepwise-communication-analysis",
            "phase": "analysis",
            "analysis_type": "stepwise_communication",
        },
    )
    run_id = run.info.run_id
    with tempfile.TemporaryDirectory(prefix="communication-output-") as temporary:
        output = Path(temporary)
        sources = {
            "probe_run_ids": args.probe_run_ids,
            "sweep_run_ids": args.sweep_run_ids,
            "independent_run_ids": args.independent_run_ids,
            "inputs": provenance,
        }
        (output / "source_runs.json").write_text(json.dumps(sources, indent=2) + "\n")
        status = "FAILED"
        try:
            analyze(probes, sweep, independent, output, args.annotate_steps)
            status = "FINISHED"
        finally:
            try:
                for path in sorted(output.iterdir()):
                    client.log_artifact(run_id, str(path), "communication")
                    if args.output_dir:
                        args.output_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, args.output_dir / path.name)
            except Exception:
                status = "FAILED"
                raise
            finally:
                client.set_terminated(run_id, status)
                print(f"MLflow analysis run: {run_id} ({status})")


if __name__ == "__main__":
    main()
