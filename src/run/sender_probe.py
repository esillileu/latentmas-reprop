"""Analysis-only entry point for saved sender latent states."""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml
from dotenv import load_dotenv

from latentmas_reprop.application.sender_probe import (
    ProbeAnalysisConfig,
    SenderProbeAnalysisUseCase,
    replace_probe_run,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze saved sender latent states")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--states", type=Path)
    source.add_argument("--source-run-id")
    parser.add_argument(
        "--source-artifact-path",
        help="Defaults to the source experiment's saved states path",
    )
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--tracking-uri")
    parser.add_argument(
        "--replace-source-run",
        action="store_true",
        help="Replace a legacy source run after copying acquisition data and traces.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument(
        "--backend", choices=["auto", "torch", "sklearn"], default="auto"
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser


def _read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if args.source_run_id and not (
        args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    ):
        raise ValueError(
            "MLflow source analysis requires --tracking-uri or MLFLOW_TRACKING_URI"
        )
    tracker = MLflowTracker(tracking_uri=args.tracking_uri)
    analysis_config = ProbeAnalysisConfig(
        seed=args.seed,
        folds=args.folds,
        permutations=args.permutations,
        backend=args.backend,
        workers=args.workers,
        c=args.c,
        max_iter=args.max_iter,
        tol=args.tol,
        batch_size=args.batch_size,
    )
    if args.replace_source_run:
        if not args.source_run_id:
            raise ValueError("--replace-source-run requires --source-run-id")
        client = mlflow.MlflowClient()
        source = client.get_run(args.source_run_id)
        if (
            client.get_experiment(source.info.experiment_id).name
            == "latentmas_receiver_trajectory"
        ):
            raise ValueError(
                "Trajectory sweeps must be preserved; analyze without --replace-source-run"
            )
        replacement_id = replace_probe_run(args.source_run_id, analysis_config)
        print(json.dumps({"replacement_run_id": replacement_id}, indent=2))
        return 0
    source_config = {}
    source_artifact = None
    experiment_name = "latentmas_sender_probe"
    with tempfile.TemporaryDirectory(prefix="latentmas-probe-") as temporary:
        if args.source_run_id:
            client = mlflow.MlflowClient()
            source_run = client.get_run(args.source_run_id)
            source_experiment = client.get_experiment(
                source_run.info.experiment_id
            ).name
            trajectory = source_experiment == "latentmas_receiver_trajectory"
            if trajectory:
                if (
                    source_run.info.status != "FINISHED"
                    or source_run.data.tags.get("phase") != "sweep"
                ):
                    raise ValueError(
                        "Sender probe requires a completed trajectory sweep"
                    )
                if args.source_config or args.source_artifact_path not in (
                    None,
                    "sender_latent_states.pt",
                ):
                    raise ValueError(
                        "Trajectory probes must use that sweep's saved states and config"
                    )
                experiment_name = source_experiment
                source_config = json.loads(
                    Path(
                        mlflow.artifacts.download_artifacts(
                            run_id=args.source_run_id,
                            artifact_path="source.json",
                            dst_path=temporary,
                        )
                    ).read_text()
                )
                if (
                    source_config["smoke"] is not False
                    or source_config["sample_count"] != 100
                ):
                    raise ValueError("Sender probe requires a full trajectory sweep")
                artifact_path = "sender_latent_states.pt"
            else:
                source_config = _read_config(args.source_config)
                artifact_path = (
                    args.source_artifact_path or "probe/sender_latent_states.pt"
                )
            states_path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=args.source_run_id,
                    artifact_path=artifact_path,
                    dst_path=temporary,
                )
            )
            source_artifact = artifact_path
            if not source_config:
                try:
                    config_path = Path(
                        mlflow.artifacts.download_artifacts(
                            run_id=args.source_run_id,
                            artifact_path="config/resolved_config.yaml",
                            dst_path=temporary,
                        )
                    )
                    source_config = _read_config(config_path)
                except Exception:
                    source_config = {}
        else:
            source_config = _read_config(args.source_config)
            states_path = args.states
        if states_path is None or not states_path.is_file():
            raise FileNotFoundError(f"sender states not found: {states_path}")
        result = SenderProbeAnalysisUseCase(tracker).execute(
            states_path,
            analysis_config,
            source_run_id=args.source_run_id,
            source_artifact_path=source_artifact,
            source_config=source_config,
            experiment_name=experiment_name,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
