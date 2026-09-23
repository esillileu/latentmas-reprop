"""Analysis-only entry point for saved sender latent states."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml

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
        "--source-artifact-path", default="probe/sender_latent_states.pt"
    )
    parser.add_argument("--source-config", type=Path)
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
    args = build_parser().parse_args(argv)
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
        replacement_id = replace_probe_run(args.source_run_id, analysis_config)
        print(json.dumps({"replacement_run_id": replacement_id}, indent=2))
        return 0
    source_config = _read_config(args.source_config)
    source_artifact = None
    with tempfile.TemporaryDirectory(prefix="latentmas-probe-") as temporary:
        if args.source_run_id:
            states_path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=args.source_run_id,
                    artifact_path=args.source_artifact_path,
                    dst_path=temporary,
                )
            )
            source_artifact = args.source_artifact_path
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
            states_path = args.states
        if states_path is None or not states_path.is_file():
            raise FileNotFoundError(f"sender states not found: {states_path}")
        result = SenderProbeAnalysisUseCase(MLflowTracker()).execute(
            states_path,
            analysis_config,
            source_run_id=args.source_run_id,
            source_artifact_path=source_artifact,
            source_config=source_config,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
