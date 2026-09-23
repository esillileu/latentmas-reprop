"""Tracking boundary for derived sender-probe analyses."""

import contextlib
import json
import tempfile
import traceback
from pathlib import Path
from typing import Any

import torch

from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...infrastructure.paths.resolver import get_git_commit_hash
from .analysis import ProbeAnalysisConfig, analyze_sender_states


class SenderProbeAnalysisUseCase:
    def __init__(self, tracker_port: ExperimentTrackerPort | None = None) -> None:
        self.tracker_port = tracker_port

    def execute(
        self,
        states_path: Path,
        config: ProbeAnalysisConfig,
        *,
        source_run_id: str | None = None,
        source_artifact_path: str | None = None,
        source_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = torch.load(states_path, map_location="cpu", weights_only=False)
        tracker = self.tracker_port
        if tracker:
            tracker.start_run(
                experiment_name="latentmas_sender_probe",
                run_name="sender_probe_analysis",
                tags={
                    "experiment_type": "sender_probe",
                    "source_run_id": source_run_id or "local_file",
                    "git_commit": get_git_commit_hash(),
                },
            )
        try:
            results, statistics = analyze_sender_states(payload, config)
            source = source_artifact_path or str(states_path.resolve())
            if tracker:
                analysis = results["analysis"]
                tracker.log_params(
                    {
                        "source_run_id": source_run_id or "local_file",
                        "source_artifact": source,
                        "git_commit": get_git_commit_hash(),
                        "seed": config.seed,
                        "folds": config.folds,
                        "permutations": config.permutations,
                        "backend": analysis["resolved_backend"],
                        "solver": analysis["solver"],
                        "workers": analysis["resolved_workers"],
                        "C": config.c,
                        "max_iter": config.max_iter,
                        "tol": config.tol,
                        "batch_size": config.batch_size,
                    }
                )
                tracker.log_dict(results, "probe/results.json")
                tracker.log_dict(statistics, "probe/null_statistics.json")
                tracker.log_dict(source_config or {}, "source/resolved_config.yaml")
                tracker.log_metrics(
                    {
                        f"accuracy/{cell['representation']}/step_{cell['step']}": cell[
                            "observed_oof_accuracy"
                        ]
                        for cell in results["cells"]
                    }
                )
                tracker.end_run("FINISHED")
            output_dir = Path(tempfile.gettempdir()) / "latentmas_sender_probe"
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )
            return results
        except Exception:
            if tracker:
                with contextlib.suppress(Exception):
                    tracker.log_params(
                        {"failure_traceback": traceback.format_exc()[:500]}
                    )
                    tracker.end_run("FAILED")
            raise
