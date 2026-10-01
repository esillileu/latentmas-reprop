"""Reanalyze saved Receiver MLflow artifacts without loading a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from mlflow.tracking import MlflowClient

from latentmas_reprop.application.receiver_acquisition.analysis import (
    analyze_records,
    read_records,
    select_receiver_runs,
)
from latentmas_reprop.application.receiver_acquisition.markdown import render_markdown
from latentmas_reprop.application.receiver_acquisition.report import (
    export_tables,
    sender_cells,
)
from latentmas_reprop.application.sender_probe.diagnostics import analyze_saved_features
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
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sender-run-id",
        action="append",
        default=[],
        help="Select saved Sender probe runs; repeat as needed.",
    )
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    client = MlflowClient()
    experiments = client.search_experiments(
        filter_string="name = 'latentmas_receiver_acquisition'"
    )
    if len(experiments) != 1:
        raise ValueError("Expected one latentmas_receiver_acquisition experiment")
    candidates = []
    records_by_id = {}
    probes_by_id = {}
    probe_sources = {}
    probe_statistics = {}
    sender_experiments = client.search_experiments(
        filter_string="name = 'latentmas_sender_probe'"
    )
    for run in client.search_runs(
        [e.experiment_id for e in sender_experiments], max_results=1000
    ):
        probe_artifacts = {
            item.path for item in client.list_artifacts(run.info.run_id, "probe")
        }
        if {"probe/results.json", "probe/null_statistics.json"} <= probe_artifacts:
            probe_cache = paths.get_cache_layer_dir(
                f"receiver_acquisition/mlflow/{run.info.run_id}"
            )
            payloads = [
                json.loads(
                    Path(
                        client.download_artifacts(
                            run.info.run_id, name, dst_path=str(probe_cache)
                        )
                    ).read_text()
                )
                for name in ("probe/results.json", "probe/null_statistics.json")
            ]
            source_id = run.data.params.get("source_run_id")
            metadata = (
                client.get_run(source_id).data.params if source_id else run.data.params
            )
            model = metadata.get("model")
            steps = metadata.get("latent_steps")
            if model is None or steps is None:
                raise ValueError(
                    f"Missing Sender model/steps metadata: {run.info.run_id}"
                )
            probe_sources[run.info.run_id] = source_id or run.info.run_id
            probe_statistics[run.info.run_id] = payloads[1]
            probes_by_id[run.info.run_id] = [
                {
                    "run_id": run.info.run_id,
                    "model": model,
                    "latent_steps": int(steps),
                    **cell,
                }
                for cell in sender_cells(*payloads)
            ]
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
        conditions = analyze_records(
            records_by_id[candidate["run_id"]],
            permutations=args.permutations,
            seed=args.seed,
        )
        if not all(
            condition in conditions
            for condition in ("drop", "latent_only/own", "latent_only/cross")
        ):
            raise ValueError(
                f"Selected Receiver run has missing conditions: {candidate['run_id']}"
            )
        runs.append(
            {
                "run_id": candidate["run_id"],
                "model": model,
                "latent_steps": steps,
                "sample_count": candidate["sample_count"],
                "conditions": conditions,
            }
        )
    runs.sort(key=lambda run: (run["model"], run["latent_steps"], run["run_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.md"
    probe_ids = set(args.sender_run_id) if args.sender_run_id else set(probes_by_id)
    if probe_ids - probes_by_id.keys():
        raise ValueError(
            f"Sender runs missing saved probe artifacts: {sorted(probe_ids - probes_by_id.keys())}"
        )
    probe_groups = {}
    probes = []
    for run_id in sorted(probe_ids):
        cells = probes_by_id[run_id]
        key = (cells[0]["model"], cells[0]["latent_steps"])
        if key in probe_groups:
            raise ValueError(
                f"Ambiguous Sender runs for {key}: {probe_groups[key]}, {run_id}; use --sender-run-id"
            )
        probe_groups[key] = run_id
        probes.extend(cells)
    robustness, geometry = [], []
    for run_id in sorted(probe_ids):
        source_id = probe_sources[run_id]
        artifact = "probe/sender_latent_states.pt"
        available = {item.path for item in client.list_artifacts(source_id, "probe")}
        if artifact not in available:
            for cell in probes_by_id[run_id]:
                cell.update(effect_ci_low=None, effect_ci_high=None)
            continue
        cache = paths.get_cache_layer_dir(f"receiver_acquisition/mlflow/{source_id}")
        state_path = client.download_artifacts(source_id, artifact, dst_path=str(cache))
        payload = torch.load(state_path, map_location="cpu", weights_only=True)
        print(f"Analyzing saved Sender features: {run_id}", flush=True)
        probe_rows, geometry_rows = analyze_saved_features(
            payload,
            probe_statistics[run_id],
            probes_by_id[run_id],
            seed=args.seed,
            bootstrap_count=args.permutations,
        )
        robustness.extend(probe_rows)
        geometry.extend(geometry_rows)
    probes.sort(
        key=lambda row: (
            row["model"],
            row["latent_steps"],
            row["representation"],
            row["step"],
        )
    )
    export_tables(args.output_dir, runs, probes, robustness, geometry)
    summary_path.write_text(
        render_markdown(runs, probes, robustness, geometry), encoding="utf-8"
    )
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
