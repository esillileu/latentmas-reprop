"""Reanalyze an MLflow preflight run without model inference or local cache inputs."""

import argparse
import json
import tempfile
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_compute_preflight.analysis import export
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--target-accuracies", default="0.5,0.7,0.9")
    args = parser.parse_args(argv)
    MLflowTracker()
    client = MlflowClient()
    run = client.get_run(args.run_id)
    if (
        run.info.status != "FINISHED"
        or run.data.tags.get("experiment_type") != "receiver_compute_preflight"
    ):
        raise ValueError("Expected a FINISHED receiver compute preflight run")
    output = (
        args.output_dir or Path("artifacts/receiver_compute_preflight") / args.run_id
    )
    with tempfile.TemporaryDirectory(prefix="preflight-analysis-") as tmp:
        records_path = client.download_artifacts(
            args.run_id, "results/sample_results.jsonl", dst_path=tmp
        )
        config_path = client.download_artifacts(
            args.run_id, "results/resolved_config.yaml", dst_path=tmp
        )
        records = [
            json.loads(line)
            for line in Path(records_path).read_text().splitlines()
            if line.strip()
        ]
        config = yaml.safe_load(Path(config_path).read_text())
    if args.bootstrap_count < 1:
        parser.error("bootstrap count must be positive")
    try:
        targets = [float(x) for x in args.target_accuracies.split(",")]
    except ValueError:
        parser.error("target accuracies must be comma-separated numbers")
    if not targets or any(not 0 <= x <= 1 for x in targets):
        parser.error("target accuracies must be between zero and one")
    config.update(bootstrap_count=args.bootstrap_count, target_accuracies=targets)
    export(output, records, config)
    (output / "analysis_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (output / "source.json").write_text(
        json.dumps(
            {"run_id": args.run_id, "artifact_uri": run.info.artifact_uri}, indent=2
        )
    )
    for name in (
        "summary.md",
        "metrics.json",
        "sample_matrix.csv",
        "budget_curves.csv",
        "bootstrap_statistics.json",
        "source.json",
        "analysis_config.yaml",
    ):
        client.log_artifact(args.run_id, str(output / name), artifact_path="analysis")
    print(f"Analysis: {output}")


if __name__ == "__main__":
    main()
