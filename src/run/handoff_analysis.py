"""Analyze saved Latent Handoff Pilot MLflow artifacts without inference."""

import argparse
from pathlib import Path

from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_reasoning.collection import analyze_runs
from latentmas_reprop.application.receiver_reasoning.mlflow import select_runs
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version-tag",
        required=True,
        help="Select only runs with this exact MLflow version_tag value.",
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
    run_ids = select_runs(
        client,
        version_tag=args.version_tag,
        experiment_name=args.experiment_name,
    )
    output = paths.resolve(args.output_dir)
    analyze_runs(
        client,
        paths,
        run_ids,
        output,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    print(f"Report index: {output / 'summary.md'}")


if __name__ == "__main__":
    main()
