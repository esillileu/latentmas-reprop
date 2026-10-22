"""Reanalyze an MLflow preflight run without model inference or local cache inputs."""

import argparse
import json
import re
import tempfile
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_compute_preflight.analysis import export
from latentmas_reprop.application.receiver_compute_preflight.plots import (
    generate_all_plots,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def sanitize_model_name(name: str) -> str:
    """Sanitize model name for filesystem directory use."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def find_runs_by_model(
    client: MlflowClient, experiment_name: str, model_query: str
) -> list[str]:
    """Find latest finished run ID(s) matching model query in the given experiment."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1000,
    )
    preflight_runs = [
        r
        for r in runs
        if r.data.tags.get("experiment_type") == "receiver_compute_preflight"
    ]
    if not preflight_runs:
        raise ValueError(
            f"No FINISHED receiver_compute_preflight runs found in {experiment_name}"
        )

    # Map each model_name to its most recent FINISHED run
    latest_by_model: dict[str, str] = {}
    for r in preflight_runs:
        m = r.data.tags.get("model_name") or r.data.params.get("model_name")
        if m and m not in latest_by_model:
            latest_by_model[m] = r.info.run_id

    if model_query.lower() == "all":
        return list(latest_by_model.values())

    query_lower = model_query.lower()

    # Priority 1: Exact full or sanitized match
    exact_matches = [
        (m, rid)
        for m, rid in latest_by_model.items()
        if m.lower() == query_lower or sanitize_model_name(m).lower() == query_lower
    ]
    if exact_matches:
        return [rid for _, rid in exact_matches]

    # Priority 2: Basename exact match (e.g. 'qwen3-4b' or 'qwen3_4b')
    base_matches = [
        (m, rid)
        for m, rid in latest_by_model.items()
        if m.split("/")[-1].lower() == query_lower
        or sanitize_model_name(m.split("/")[-1]).lower() == query_lower
    ]
    if base_matches:
        return [rid for _, rid in base_matches]

    # Priority 3: Word/size boundary match (e.g. '4b' matches 'Qwen3-4B' but not '14B')
    pattern = re.compile(rf"(?<!\d){re.escape(query_lower)}(?!\d)")
    boundary_matches = [
        (m, rid) for m, rid in latest_by_model.items() if pattern.search(m.lower())
    ]
    if boundary_matches:
        return [rid for _, rid in boundary_matches]

    # Priority 4: General substring match
    sub_matches = [
        (m, rid) for m, rid in latest_by_model.items() if query_lower in m.lower()
    ]
    if sub_matches:
        return [rid for _, rid in sub_matches]

    avail = sorted(latest_by_model.keys())
    raise ValueError(
        f"No finished preflight run found matching model {model_query!r}. "
        f"Available models: {avail}"
    )


def analyze_run(
    client: MlflowClient,
    run_id: str,
    output_dir: Path | None = None,
    bootstrap_count: int = 2000,
    target_accuracies: list[float] | None = None,
    generate_plots: bool = True,
) -> Path:
    """Analyze a single preflight run and export artifacts to model directory."""
    run = client.get_run(run_id)
    if (
        run.info.status != "FINISHED"
        or run.data.tags.get("experiment_type") != "receiver_compute_preflight"
    ):
        raise ValueError(
            f"Expected a FINISHED receiver compute preflight run: {run_id}"
        )

    with tempfile.TemporaryDirectory(prefix="preflight-analysis-") as tmp:
        records_path = client.download_artifacts(
            run_id, "results/sample_results.jsonl", dst_path=tmp
        )
        config_path = client.download_artifacts(
            run_id, "results/resolved_config.yaml", dst_path=tmp
        )
        records = [
            json.loads(line)
            for line in Path(records_path).read_text().splitlines()
            if line.strip()
        ]
        config = yaml.safe_load(Path(config_path).read_text())

    model_name = (
        config.get("model_name")
        or run.data.tags.get("model_name")
        or run.data.params.get("model_name")
        or run_id
    )
    model_dir = sanitize_model_name(model_name)
    output = output_dir or (Path("artifacts/receiver_compute_preflight") / model_dir)
    output.mkdir(parents=True, exist_ok=True)

    targets = target_accuracies or [0.5, 0.7, 0.9]
    config.update(bootstrap_count=bootstrap_count, target_accuracies=targets)
    metrics = export(output, records, config)

    if generate_plots:
        generate_all_plots(output, metrics, model_name=model_name)
        summary_path = output / "summary.md"
        if summary_path.exists():
            plot_section = [
                "",
                "## plots",
                "",
                "- ![Budget Accuracy Curve](budget_accuracy_curve.png)",
                "- ![Pairing Gain Curve](pairing_gain_curve.png)",
                "- ![Latency Accuracy Tradeoff](latency_accuracy_tradeoff.png)",
                "- ![Tokens Cost Curve](tokens_cost_curve.png)",
                "",
            ]
            content = summary_path.read_text(encoding="utf-8")
            if "## plots" not in content:
                summary_path.write_text(
                    content + "\n".join(plot_section), encoding="utf-8"
                )

    (output / "analysis_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    (output / "source.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "artifact_uri": run.info.artifact_uri,
                "model": model_name,
            },
            indent=2,
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
        "budget_accuracy_curve.png",
        "pairing_gain_curve.png",
        "latency_accuracy_tradeoff.png",
        "tokens_cost_curve.png",
    ):
        file_path = output / name
        if file_path.exists():
            client.log_artifact(run_id, str(file_path), artifact_path="analysis")
    print(f"Analysis ({model_name}): {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", help="Direct MLflow run ID to analyze.")
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        help="Model name to analyze (e.g. 'Qwen/Qwen3-4B', 'Qwen3-4B', '4B', or 'all').",
    )
    parser.add_argument(
        "--experiment-name",
        default="latentmas_receiver_compute_preflight",
        help="MLflow experiment name to search when --model is specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Custom output directory. Defaults to artifacts/receiver_compute_preflight/<model_name>.",
    )
    parser.add_argument("--bootstrap-count", type=int, default=2000)
    parser.add_argument("--target-accuracies", default="0.5,0.7,0.9")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip generating graphical curve plots.",
    )
    args = parser.parse_args(argv)

    if not args.run_id and not args.model_name:
        parser.error("Either --model/--model-name or --run-id must be specified.")

    if args.bootstrap_count < 1:
        parser.error("bootstrap count must be positive")
    try:
        targets = [float(x) for x in args.target_accuracies.split(",")]
    except ValueError:
        parser.error("target accuracies must be comma-separated numbers")
    if not targets or any(not 0 <= x <= 1 for x in targets):
        parser.error("target accuracies must be between zero and one")

    MLflowTracker()
    client = MlflowClient()

    if args.run_id:
        analyze_run(
            client=client,
            run_id=args.run_id,
            output_dir=args.output_dir,
            bootstrap_count=args.bootstrap_count,
            target_accuracies=targets,
            generate_plots=not args.no_plots,
        )
    else:
        run_ids = find_runs_by_model(client, args.experiment_name, args.model_name)
        for rid in run_ids:
            out = args.output_dir if len(run_ids) == 1 else None
            analyze_run(
                client=client,
                run_id=rid,
                output_dir=out,
                bootstrap_count=args.bootstrap_count,
                target_accuracies=targets,
                generate_plots=not args.no_plots,
            )


if __name__ == "__main__":
    main()
