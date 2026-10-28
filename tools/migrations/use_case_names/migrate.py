"""One-time migration for the use case renaming.

Moves existing results to the renamed storage paths, verifies file contents and
completion state, and realigns MLflow path references. Nothing is deleted:
experiment directories are renamed in place, MLflow runs, experiments and traces
are never removed, merged or copied.

Default mode is a dry run; pass ``--execute`` to apply changes.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage.tensor_store import is_condition_complete

BASE_STEM = (
    r"ls\d+_seed-?\d+_temp[^_]+_top[^_]+_mt\d+(?:_split[^_]+)?"
    r"_[A-Za-z0-9]+(?:_vllm)?(?:_think)?(?:_realign)?"
)
USE_CASE_TOKEN_RE = re.compile(
    rf"^(?P<base>{BASE_STEM})_(?P<token>single_intervention|intervention(?!(?:_kv|_continuation)(?:_|$))|acquisition)"
    r"(?P<rest>(?:_.*)?)$"
)
TRAJECTORY_RE = re.compile(rf"^(?P<head>{BASE_STEM}.*)_trajectory(?P<tail>(?:_smoke)?)$")
TOKEN_RENAMES = {
    "single_intervention": "intervention_continuation",
    "intervention": "intervention_kv",
    "acquisition": "representation_acquisition",
}

# Local analysis/cache directories, relative to (artifacts_dir, cache_dir).
ARTIFACT_DIR_RENAMES = {
    "receiver_acquisition": "representation_acquisition",
    "receiver_trajectory": "representation_trajectory",
}
CACHE_DIR_RENAMES = {
    "evaluation/receiver_acquisition": "evaluation/representation_acquisition",
}

# One-to-one MLflow experiment renames. ``latentmas-reprop`` is intentionally
# absent: it holds benchmark and intervention runs, so it cannot be renamed
# without merging or splitting; it is preserved as is.
EXPERIMENT_RENAMES = {
    "latentmas_intervention": "latentmas/intervention/kv",
    "latentmas_single_intervention": "latentmas/intervention/continuation",
    "latentmas_receiver_acquisition": "latentmas/representation/acquisition",
    "latentmas_receiver_trajectory": "latentmas/representation/trajectory",
    "latentmas_sender_probe": "latentmas/representation/sender_probe",
}


def new_params_stem(stem: str) -> str | None:
    """Return the renamed params directory name, or None if it is unaffected."""
    match = USE_CASE_TOKEN_RE.match(stem)
    if match:
        renamed = TOKEN_RENAMES[match["token"]]
        return f"{match['base']}_{renamed}{match['rest']}"
    match = TRAJECTORY_RE.match(stem)
    if match:
        return f"{match['head']}_representation_trajectory{match['tail']}"
    return None


def new_relpath(rel: str) -> str | None:
    """Rename ``<task>/<model>/<method>/<prompt>/<params>``; None if unaffected."""
    parts = rel.strip("/").split("/")
    if len(parts) != 5:
        return None
    renamed = new_params_stem(parts[-1])
    return None if renamed is None else "/".join([*parts[:-1], renamed])


@dataclass
class DirMove:
    kind: str
    source: str
    dest: str
    status: str = "planned"
    detail: str = ""


def file_manifest(root: Path, verify: str) -> dict[str, tuple[int, str]]:
    """Relative path -> (size, sha256 or '') for every file below root."""
    manifest = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = ""
        if verify == "sha256":
            hasher = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        manifest[path.relative_to(root).as_posix()] = (path.stat().st_size, digest)
    return manifest


def completion_state(root: Path) -> dict[str, bool]:
    """Completion verdict for every directory holding a result.json."""
    return {
        result.parent.relative_to(root).as_posix(): is_condition_complete(result.parent)
        for result in sorted(root.rglob("result.json"))
    }


def plan_experiment_moves(experiments_dir: Path) -> list[DirMove]:
    moves = []
    if not experiments_dir.is_dir():
        return moves
    for task in sorted(experiments_dir.iterdir()):
        for model in sorted(p for p in task.iterdir() if p.is_dir()) if task.is_dir() else []:
            for method in sorted(p for p in model.iterdir() if p.is_dir()):
                for prompt in sorted(p for p in method.iterdir() if p.is_dir()):
                    for params in sorted(p for p in prompt.iterdir() if p.is_dir()):
                        renamed = new_params_stem(params.name)
                        if renamed is not None:
                            moves.append(
                                DirMove(
                                    "experiment",
                                    str(params),
                                    str(params.with_name(renamed)),
                                )
                            )
    return moves


def plan_local_moves(artifacts_dir: Path, cache_dir: Path) -> list[DirMove]:
    moves = []
    for base, renames in ((artifacts_dir, ARTIFACT_DIR_RENAMES), (cache_dir, CACHE_DIR_RENAMES)):
        for old, new in renames.items():
            if (base / old).is_dir():
                moves.append(DirMove("local", str(base / old), str(base / new)))
    return moves


def check_conflicts(moves: list[DirMove]) -> None:
    destinations: dict[str, str] = {}
    for move in moves:
        if Path(move.dest).exists():
            move.status = "conflict"
            move.detail = "destination already exists"
        elif move.dest in destinations:
            move.status = "conflict"
            move.detail = f"same destination as {destinations[move.dest]}"
        destinations.setdefault(move.dest, move.source)


def apply_move(move: DirMove, verify: str) -> None:
    source, dest = Path(move.source), Path(move.dest)
    before_files, before_state = file_manifest(source, verify), completion_state(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, dest)
    if file_manifest(dest, verify) != before_files or completion_state(dest) != before_state:
        os.rename(dest, source)
        move.status, move.detail = "failed", "verification failed; rolled back"
        return
    move.status = "moved"
    move.detail = f"{len(before_files)} files verified ({verify or 'size'})"


def run_filesystem(moves: list[DirMove], execute: bool, verify: str) -> None:
    check_conflicts(moves)
    for move in moves:
        if move.status == "conflict":
            continue
        if execute:
            apply_move(move, verify)
        print(f"[{move.status}] {move.source} -> {move.dest} {move.detail}")


def _tag_updates(tags: dict[str, str], experiments_dir: Path) -> dict[str, str]:
    """New tag values for a run whose rel_dir is affected by the rename."""
    rel = tags.get("rel_dir")
    renamed = new_relpath(rel) if rel else None
    if not renamed:
        return {}
    updates = {"rel_dir": renamed}
    if tags.get("run_name") == rel:
        updates["run_name"] = renamed
    data_dir = tags.get("data_dir")
    if data_dir and data_dir.endswith(rel):
        updates["data_dir"] = data_dir[: -len(rel)] + renamed
    elif data_dir:
        updates["data_dir"] = str((experiments_dir / renamed).resolve())
    return updates


def run_mlflow(
    tracking_uri: str, experiments_dir: Path, execute: bool, rename_experiments: bool
) -> dict:
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    summary = {"runs_updated": 0, "experiments_renamed": [], "experiments_skipped": []}
    experiments = client.search_experiments(view_type=ViewType.ALL)
    for experiment in experiments:
        token = None
        while True:
            runs = client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ALL,
                max_results=1000,
                page_token=token,
            )
            for run in runs:
                updates = _tag_updates(run.data.tags, experiments_dir)
                if not updates:
                    continue
                summary["runs_updated"] += 1
                print(f"[mlflow-run] {run.info.run_id} {updates}")
                if execute:
                    for key, value in updates.items():
                        client.set_tag(run.info.run_id, key, value)
                    if run.info.run_name == run.data.tags.get("rel_dir"):
                        client.update_run(run.info.run_id, name=updates["rel_dir"])
            token = runs.token
            if not token:
                break
    if rename_experiments:
        names = {e.name for e in experiments}
        for experiment in experiments:
            target = EXPERIMENT_RENAMES.get(experiment.name)
            if target is None:
                continue
            if target in names:
                summary["experiments_skipped"].append(
                    f"{experiment.name}: target {target} already exists"
                )
                print(f"[mlflow-experiment-skip] {experiment.name} -> {target} exists")
                continue
            print(f"[mlflow-experiment] {experiment.name} -> {target}")
            if execute:
                client.rename_experiment(experiment.experiment_id, target)
            summary["experiments_renamed"].append(f"{experiment.name} -> {target}")
    return summary


def main(argv: list[str] | None = None) -> int:
    resolver = PathResolver()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Apply the migration.")
    parser.add_argument("--experiments-dir", type=Path, default=resolver.experiments_dir)
    parser.add_argument("--artifacts-dir", type=Path, default=resolver.artifacts_dir)
    parser.add_argument("--cache-dir", type=Path, default=resolver.cache_dir)
    parser.add_argument("--mlflow-tracking-uri", default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    parser.add_argument("--rename-mlflow-experiments", action="store_true")
    parser.add_argument("--verify", choices=("size", "sha256"), default="size")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    print("[EXECUTE]" if args.execute else "[DRY RUN] no changes will be made")
    moves = plan_experiment_moves(args.experiments_dir) + plan_local_moves(
        args.artifacts_dir, args.cache_dir
    )
    run_filesystem(moves, args.execute, args.verify)

    mlflow_summary = None
    uri = args.mlflow_tracking_uri or (
        f"sqlite:///{resolver.mlflow_db_path.resolve()}"
        if resolver.mlflow_db_path.is_file()
        else None
    )
    if not args.skip_mlflow and uri:
        mlflow_summary = run_mlflow(
            uri, args.experiments_dir, args.execute, args.rename_mlflow_experiments
        )

    failed = [m for m in moves if m.status in {"conflict", "failed"}]
    report = {
        "executed": args.execute,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "moves": [asdict(m) for m in moves],
        "mlflow": mlflow_summary,
    }
    if args.execute:
        default_rep = args.experiments_dir.parent / "migrations" / f"use_case_names_{datetime.now():%Y%m%d_%H%M%S}.json"
        report_path = args.report or default_rep
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"report: {report_path}")
    if failed:
        print(f"{len(failed)} move(s) skipped or failed; resolve and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
