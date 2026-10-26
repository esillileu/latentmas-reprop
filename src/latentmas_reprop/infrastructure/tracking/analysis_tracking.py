"""Tracking utilities for analysis runs registered in MLflow."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .mlflow_tracker import MLflowTracker


def log_analysis_results_to_mlflow(
    tracker: "MLflowTracker",
    analysis_rel_path: str,
    output_dir: Path,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    """Register analysis run under MLflow 'latentmas-analysis' experiment and upload outputs."""
    tracker.start_run(
        experiment_name="latentmas-analysis",
        run_name=analysis_rel_path,
        tags={
            "analysis_path": analysis_rel_path,
            "data_dir": str(output_dir.resolve()),
            "rel_dir": analysis_rel_path,
        },
    )
    if params:
        tracker.log_params(params)
    if metrics:
        clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        if clean:
            tracker.log_metrics(clean)
    if output_dir.is_dir():
        for file in sorted(output_dir.rglob("*")):
            if file.is_file():
                rel = file.relative_to(output_dir)
                art_dir = str(rel.parent) if rel.parent != Path(".") else None
                tracker.log_artifact(file, artifact_path=art_dir)
    tracker.end_run(status="FINISHED")
