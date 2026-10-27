"""Analyze saved Latent Handoff Pilot MLflow artifacts without inference."""

import argparse
import contextlib
from pathlib import Path

from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_reasoning.collection import analyze_runs
from latentmas_reprop.application.receiver_reasoning.mlflow import select_runs
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import (
    MLflowTracker,
    log_analysis_results_to_mlflow,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        "--input-dir",
        action="append",
        default=[],
        help="Specify experiment directories to analyze.",
    )
    parser.add_argument(
        "--version-tag",
        default=None,
        help="Select runs with this exact MLflow version_tag value.",
    )
    parser.add_argument("--experiment-name", default="latentmas_receiver_reasoning")
    parser.add_argument("--tracking-uri")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/latent_handoff")
    )
    parser.add_argument("--bootstrap-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.bootstrap_count < 1:
        parser.error("--bootstrap-count must be positive")
    paths = get_path_resolver()
    # Reuse the repository's local SQLite / MLFLOW_TRACKING_URI policy.
    tracker = MLflowTracker(tracking_uri=args.tracking_uri)
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    if args.experiment_dir:
        run_ids = []
        exp = client.get_experiment_by_name(args.experiment_name)
        if exp:
            for ed in args.experiment_dir:
                edir_path = paths.get_experiment_dir(ed, create=False)
                edir = str(edir_path.resolve())
                runs = client.search_runs(
                    [exp.experiment_id],
                    filter_string=f"tags.data_dir = '{edir}'",
                    max_results=1,
                )
                if not runs:
                    with contextlib.suppress(ValueError):
                        rel_dir = str(edir_path.relative_to(paths.experiments_dir))
                        runs = client.search_runs(
                            [exp.experiment_id],
                            filter_string=f"tags.rel_dir = '{rel_dir}'",
                            max_results=1,
                        )
                if not runs:
                    runs = client.search_runs(
                        [exp.experiment_id],
                        filter_string=f"tags.rel_dir = '{ed}'",
                        max_results=1,
                    )
                for r in runs:
                    run_ids.append(r.info.run_id)
        if not run_ids:
            raise FileNotFoundError(
                f"No runs found matching experiment directory: {args.experiment_dir}"
            )
    elif args.version_tag:
        run_ids = select_runs(
            client,
            version_tag=args.version_tag,
            experiment_name=args.experiment_name,
        )
    else:
        # Default: select all finished runs in experiment
        exp = client.get_experiment_by_name(args.experiment_name)
        runs = (
            client.search_runs(
                [exp.experiment_id],
                filter_string="attributes.status = 'FINISHED'",
                max_results=100,
            )
            if exp
            else []
        )
        run_ids = [r.info.run_id for r in runs]
    output = paths.resolve(args.output_dir)
    analyze_runs(
        client,
        paths,
        run_ids,
        output,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    log_analysis_results_to_mlflow(
        tracker,
        "latent_handoff",
        output,
        params={"seed": args.seed, "bootstrap_count": args.bootstrap_count},
    )
    print(f"Report index: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
