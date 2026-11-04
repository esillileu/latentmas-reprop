"""Backfill completed benchmark runs from storage into MLflow latentmas/benchmark experiment."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import mlflow

from latentmas_reprop.application.benchmark.artifacts import trace_benchmark_sample
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def find_completed_runs(base_dir: Path) -> list[Path]:
    runs = []
    summaries = []
    for root, dirs, files in os.walk(base_dir):
        # Never descend into per-sample dirs; skip non-benchmark (intervention) runs
        dirs[:] = [d for d in dirs if d != "samples" and "intervention" not in d]
        if "summary.json" in files:
            summaries.append(Path(root) / "summary.json")
    for summary_path in summaries:
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            count = data.get("sample_count") or data.get("total_samples") or 0
            if count >= 1319 and (summary_path.parent / "samples.jsonl").exists():
                runs.append(summary_path.parent)
        except Exception:
            continue
    return sorted(runs)


def backfill_run(
    run_dir: Path,
    tracker: MLflowTracker,
    client: mlflow.tracking.MlflowClient,
    experiment_name: str,
) -> None:
    summary_path = run_dir / "summary.json"
    samples_path = run_dir / "samples.jsonl"
    config_path = run_dir / "config.json"

    with open(summary_path, encoding="utf-8") as f:
        summary_data = json.load(f)

    cfg = summary_data.get("config", {})
    if config_path.exists() and not cfg:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)

    metrics_data = summary_data.get("metrics", {})
    rel_path = str(run_dir.relative_to(Path("storage/experiments")))
    # Folder names repeat across models/methods, so the full relpath is the run name
    run_name = rel_path

    # Check if run already exists in experiment
    exp = client.get_experiment_by_name(experiment_name)
    if exp:
        existing = client.search_runs(
            exp.experiment_id,
            filter_string=f"tags.`mlflow.runName` = '{run_name}'",
            max_results=1,
        )
        if existing and existing[0].info.status == "FINISHED":
            print(f"[Skip] Run '{run_name}' already exists in {experiment_name}.")
            return

    print(f"[*] Backfilling '{run_name}' ({rel_path})...")

    # Construct args namespace for trace_benchmark_sample
    model_name = (
        cfg.get("model")
        or cfg.get("model_name")
        or metrics_data.get("model", "unknown")
    )
    method = cfg.get("method") or metrics_data.get("method", "unknown")
    task = cfg.get("task") or metrics_data.get("task", "gsm8k")
    split = cfg.get("split") or metrics_data.get("split", "test")

    args = SimpleNamespace(
        model_name=model_name,
        method=method,
        task=task,
        split=split,
        temperature=cfg.get("temperature", 0.6),
        top_p=cfg.get("top_p", 0.95),
        latent_steps=cfg.get("latent_steps", 0),
        seed=cfg.get("seed", 42),
    )

    tags = {
        "experiment_type": "benchmark",
        "task": str(task),
        "model": str(model_name),
        "method": str(method),
        "split": str(split),
        "seed": str(args.seed),
        "rel_dir": rel_path,
    }
    if "latent_steps" in cfg:
        tags["latent_steps"] = str(cfg["latent_steps"])

    tracker.start_run(
        experiment_name=experiment_name,
        run_name=run_name,
        tags=tags,
    )

    # 1. Log parameters
    tracker.log_params(
        {
            "model": str(model_name),
            "task": str(task),
            "method": str(method),
            "split": str(split),
            "temperature": str(args.temperature),
            "top_p": str(args.top_p),
            "latent_steps": str(args.latent_steps),
            "seed": str(args.seed),
            "max_samples": str(summary_data.get("sample_count", 1319)),
        }
    )

    # 2. Upload sample traces
    print("    Tracing 1,319 samples...")
    sample_idx = 1
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            trace_benchmark_sample(tracker, args, sample_idx, rec)
            sample_idx += 1

    # Flush tracing before ending run
    mlflow.flush_trace_async_logging()

    # 3. Log metrics
    clean_metrics = {}
    for k, v in metrics_data.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            clean_metrics[k] = float(v)
    if clean_metrics:
        tracker.log_metrics(clean_metrics)

    # 4. Log artifacts
    tracker.log_artifact(summary_path, artifact_path="results")
    if config_path.exists():
        tracker.log_artifact(config_path, artifact_path="config")

    tracker.end_run(status="FINISHED")
    print(f"    [Done] '{run_name}' backfilled successfully.")


def main():
    base_dir = Path("storage/experiments/gsm8k")
    experiment_name = "latentmas/benchmark"

    completed = find_completed_runs(base_dir)
    print(f"Found {len(completed)} completed benchmark runs in {base_dir}")

    client = mlflow.tracking.MlflowClient()
    tracker = MLflowTracker()

    for idx, rdir in enumerate(completed, start=1):
        print(f"\n--- [{idx}/{len(completed)}] ---")
        backfill_run(rdir, tracker, client, experiment_name)


if __name__ == "__main__":
    main()
