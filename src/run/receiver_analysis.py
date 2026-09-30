"""Reanalyze saved Receiver MLflow artifacts without loading a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_acquisition.analysis import (
    analyze_records,
    read_records,
    render_markdown,
    select_receiver_runs,
)
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = get_path_resolver()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=paths.root / "artifacts" / "receiver_acquisition",
    )
    parser.add_argument(
        "--canonical-run-id",
        action="append",
        default=[],
        help="Designate a Receiver run for its model and latent step; repeat as needed.",
    )
    args = parser.parse_args()
    client = MlflowClient()
    experiments = client.search_experiments(
        filter_string="name = 'latentmas_receiver_acquisition'"
    )
    if len(experiments) != 1:
        raise ValueError("Expected one latentmas_receiver_acquisition experiment")
    candidates = []
    records_by_id = {}
    reference_verified = False
    for run in client.search_runs([experiments[0].experiment_id], max_results=1000):
        artifacts = {
            item.path for item in client.list_artifacts(run.info.run_id, "results")
        }
        if "results/sample_results.jsonl" not in artifacts:
            continue
        cache_dir = paths.get_cache_layer_dir(
            f"receiver_acquisition/mlflow/{run.info.run_id}"
        )
        path = Path(
            client.download_artifacts(
                run.info.run_id, "results/sample_results.jsonl", dst_path=str(cache_dir)
            )
        )
        records = read_records(path)
        if not records:
            continue
        model = records[0]["model"]
        steps = records[0]["latent_steps"]
        condition_counts = Counter(
            "drop"
            if record["condition"] == "drop"
            else f"{record['context_mode']}/{record['condition']}"
            for record in records
            if record["error"] is None
        )
        candidates.append(
            {
                "run_id": run.info.run_id,
                "model": model,
                "latent_steps": steps,
                "sample_count": len({record["sample_index"] for record in records}),
                "condition_counts": condition_counts,
                "designated": run.data.tags.get("receiver_analysis_canonical")
                == "true",
            }
        )
        records_by_id[run.info.run_id] = records
    selected = select_receiver_runs(candidates, set(args.canonical_run_id))
    runs = []
    for candidate in selected:
        model = candidate["model"]
        steps = candidate["latent_steps"]
        conditions = analyze_records(records_by_id[candidate["run_id"]])
        if not all(
            condition in conditions
            for condition in ("drop", "latent_only/own", "latent_only/cross")
        ):
            raise ValueError(
                f"Selected Receiver run has missing conditions: {candidate['run_id']}"
            )
        if model == "Qwen/Qwen3-4B" and steps == 20:
            expected = {
                "drop": ({3: 100}, 0.10),
                "latent_only/own": ({1: 80, 7: 20}, 0.20),
                "latent_only/cross": ({1: 79, 7: 21}, 0.19),
            }
            for condition, (counts, accuracy) in expected.items():
                values = conditions[condition]
                actual = {
                    int(digit): entry["count"]
                    for digit, entry in values["prediction_distribution"].items()
                    if entry["count"]
                }
                assert actual == counts, (condition, actual)
                assert abs(values["accuracy"] - accuracy) < 1e-12
            assert (
                abs(
                    conditions["latent_only/own"]["probability_delta_vs_drop"]
                    + 0.004380573949310929
                )
                < 1e-10
            )
            reference_verified = True
            print("Verified Qwen3-4B / steps=20 against saved full-sample results")
        runs.append(
            {
                "run_id": candidate["run_id"],
                "model": model,
                "latent_steps": steps,
                "sample_count": candidate["sample_count"],
                "conditions": conditions,
            }
        )
    if not reference_verified:
        raise ValueError("Qwen3-4B / steps=20 reference run was not found")
    runs.sort(key=lambda run: (run["model"], run["latent_steps"], run["run_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(render_markdown(runs), encoding="utf-8")
    for run in runs:
        model_name = run["model"].split("/")[-1]
        run_dir = (
            args.output_dir
            / "runs"
            / model_name
            / f"steps_{run['latent_steps']}"
            / run["run_id"]
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "diagnostics.json").write_text(
            json.dumps(run, indent=2) + "\n", encoding="utf-8"
        )
    selected_ids = {run["run_id"] for run in runs}
    for stale in (args.output_dir / "runs").glob("*/*/*/diagnostics.json"):
        if stale.parent.name not in selected_ids:
            stale.unlink()
            for parent in (
                stale.parent,
                stale.parent.parent,
                stale.parent.parent.parent,
            ):
                if not any(parent.iterdir()):
                    parent.rmdir()
    print(f"Analyzed {len(runs)} Receiver runs: {summary_path}")


if __name__ == "__main__":
    main()
