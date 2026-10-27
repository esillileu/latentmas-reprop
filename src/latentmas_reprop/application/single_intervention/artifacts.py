"""Artifacts management and summary reporting for single intervention analysis."""

import contextlib
import json
from pathlib import Path
from typing import Any

from ...infrastructure.storage import save_json_atomic


def build_summary_dict(
    records: list[dict[str, Any]],
    conditions: list[str],
    total_samples: int,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics and counts across all completed samples and conditions."""
    metrics: dict[str, Any] = {
        "sample_count": total_samples,
        "total": total_samples,
    }
    cond_stats: dict[str, Any] = {}
    for cond in conditions:
        cond_records = [r for r in records if r.get("condition") == cond]
        count = len(cond_records)
        correct = sum(1 for r in cond_records if r.get("correct") is True)
        acc = correct / max(1, count)
        exhausted = sum(1 for r in cond_records if r.get("budget_exhausted") is True)
        avg_len = (
            sum(r.get("generated_length", 0) for r in cond_records) / max(1, count)
        )
        cond_stats[cond] = {
            "completed": count,
            "correct": correct,
            "accuracy": acc,
            "budget_exhausted": exhausted,
            "exhausted_rate": exhausted / max(1, count),
            "avg_generated_length": avg_len,
        }
        metrics[f"{cond}_accuracy"] = acc
        metrics[f"{cond}_correct"] = correct
        metrics[f"{cond}_completed"] = count

    return {
        "status": "complete",
        "sample_count": total_samples,
        "conditions": conditions,
        "condition_statistics": cond_stats,
        "metrics": metrics,
        "config": config_dict or {},
    }


def save_experiment_summary(
    exp_dir: Path,
    summary_data: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Save summary.json and samples.jsonl atomically at the experiment root."""
    save_json_atomic(exp_dir / "summary.json", summary_data)
    jsonl_path = exp_dir / "samples.jsonl"
    tmp_jsonl = jsonl_path.with_name(f"{jsonl_path.name}.tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_jsonl.replace(jsonl_path)


def log_to_mlflow_tracker(
    tracker: Any,
    exp_dir: Path,
    exp_relpath: str,
    git_hash: str,
    args: Any,
    summary_data: dict[str, Any],
) -> None:
    """Log lightweight summary metrics, parameters, tags, and small artifacts to MLflow."""
    if not tracker:
        return
    exp_name = getattr(
        args, "tracking_experiment_name", "latentmas_single_intervention"
    )
    with contextlib.suppress(Exception):
        tracker.start_run(
            experiment_name=exp_name,
            run_name=exp_relpath,
            tags={
                "data_dir": str(exp_dir.resolve()),
                "rel_dir": exp_relpath,
                "git_commit": git_hash,
                "run_name": exp_relpath,
            },
        )
        # Log config params
        cfg_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        tracker.log_params(
            {
                k: str(v)
                for k, v in cfg_dict.items()
                if isinstance(v, (int, float, bool, str))
            }
        )
        # Log summary metrics
        metrics = summary_data.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                tracker.log_metric(k, float(v))

        # Log lightweight artifacts (never large .pt files!)
        if (exp_dir / "summary.json").is_file():
            tracker.log_artifact(str(exp_dir / "summary.json"))
        if (exp_dir / "config.json").is_file():
            tracker.log_artifact(str(exp_dir / "config.json"))
        if (exp_dir / "samples.jsonl").is_file():
            tracker.log_artifact(str(exp_dir / "samples.jsonl"))

        tracker.end_run("FINISHED")
