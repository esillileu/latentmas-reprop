"""Validate and plot saved sender/receiver metrics only; never run inference."""

import argparse
import contextlib
import csv
import json
from pathlib import Path

from latentmas_reprop.domain.models.use_cases import (
    REPRESENTATION_TRAJECTORY,
    mlflow_experiment_name,
)

from .stepwise_communication_metrics import communication_metrics, receiver_parity
from .stepwise_communication_plots import plot_metrics

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

    from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
    from latentmas_reprop.infrastructure.tracking.analysis_tracking import (
        log_analysis_results_to_mlflow,
    )
    from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker

    from .stepwise_communication_sources import load_inputs

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-dir", "--input-dir", action="append", default=[])
    for flag in ("--probe-run-ids", "--sweep-run-ids", "--independent-run-ids"):
        parser.add_argument(flag, nargs="+", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/stepwise_communication"),
        help="Local export directory for derived artifacts",
    )
    parser.add_argument("--annotate-steps", action="store_true")
    args = parser.parse_args(argv)

    tracker = MLflowTracker(tracking_uri=args.tracking_uri)
    client = MlflowClient(tracking_uri=tracker.tracking_uri)

    paths = get_path_resolver()
    probe_ids = list(args.probe_run_ids)
    sweep_ids = list(args.sweep_run_ids)
    independent_ids = list(args.independent_run_ids)

    if args.experiment_dir:
        exp = client.get_experiment_by_name(
            mlflow_experiment_name(REPRESENTATION_TRAJECTORY)
        )
        matched_any = False
        if exp:
            for ed in args.experiment_dir:
                edir_path = paths.get_experiment_dir(ed, create=False)
                edir = str(edir_path.resolve())
                rel_dir = ed
                with contextlib.suppress(ValueError):
                    rel_dir = str(edir_path.relative_to(paths.experiments_dir))
                filt = f"tags.data_dir = '{edir}' or tags.rel_dir = '{rel_dir}' or tags.rel_dir = '{ed}'"
                runs = client.search_runs(
                    [exp.experiment_id], filter_string=filt, max_results=10
                )
                for r in runs:
                    matched_any = True
                    phase = r.data.tags.get("phase")
                    if phase == "sweep":
                        sweep_ids.append(r.info.run_id)
                    elif phase == "sender_probe":
                        probe_ids.append(r.info.run_id)
                    elif phase == "independent":
                        independent_ids.append(r.info.run_id)
        if not matched_any:
            raise FileNotFoundError(
                f"No trajectory runs found matching experiment directory: {args.experiment_dir}"
            )

    probes, sweep, independent, provenance = load_inputs(
        client, probe_ids, sweep_ids, independent_ids
    )

    output = paths.resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "probe_run_ids": probe_ids,
        "sweep_run_ids": sweep_ids,
        "independent_run_ids": independent_ids,
        "inputs": provenance,
    }
    (output / "source_runs.json").write_text(json.dumps(sources, indent=2) + "\n")
    analyze(probes, sweep, independent, output, args.annotate_steps)
    analysis_name = "stepwise_communication"
    log_analysis_results_to_mlflow(
        tracker,
        analysis_name,
        output,
        params={"annotate_steps": args.annotate_steps},
    )
    print(f"MLflow analysis recorded under latentmas-analysis for: {output}")


if __name__ == "__main__":
    main()
