"""Preserve original run files and publish immutable run references."""

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .files import (
    file_hash,
    manifest,
    publish_json,
    rename_new,
    require_safe_path,
    write_json,
)


def _copy_run_files(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    for item in source.iterdir():
        if item.name == "samples":
            continue
        if item.is_symlink():
            raise ValueError(f"Symbolic link in source: {item}")
        if item.is_dir():
            shutil.copytree(item, destination / item.name, symlinks=True)
        elif item.is_file():
            shutil.copy2(item, destination / item.name)
        else:
            raise ValueError(f"Unsupported source file: {item}")


def _run_manifest(source: Path) -> dict[str, str]:
    result = {}
    for item in source.iterdir():
        if item.name == "samples":
            continue
        if item.is_symlink():
            raise ValueError(f"Symbolic link in source: {item}")
        if item.is_file():
            result[item.name] = file_hash(item)
        elif item.is_dir():
            result.update({f"{item.name}/{k}": v for k, v in manifest(item).items()})
        else:
            raise ValueError(f"Unsupported source file: {item}")
    return result


def publish_run(source: Path, destination: Path, data: dict[str, Any]) -> str:
    before = _run_manifest(source)
    if destination.exists():
        expected_files = {"migration.json", *(f"legacy/{name}" for name in before)}
        if "config" in data:
            expected_files.add("config.json")
        expected_files.update(
            name for name in ("summary.json", "samples.jsonl") if name in before
        )
        if (
            not (destination / "legacy").is_dir()
            or set(manifest(destination)) != expected_files
            or _run_manifest(source) != before
            or not (destination / "migration.json").is_file()
            or json.loads((destination / "migration.json").read_text()) != data
            or manifest(destination / "legacy") != before
            or (
                "config" in data
                and (
                    not (destination / "config.json").is_file()
                    or json.loads((destination / "config.json").read_text())
                    != data["config"]
                )
            )
            or any(
                (destination / name).is_file() != (name in before)
                or (name in before and file_hash(destination / name) != before[name])
                for name in ("summary.json", "samples.jsonl")
            )
        ):
            raise ValueError(f"Existing run conflicts: {destination}")
        return "reused"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".run-", dir=destination.parent))
    try:
        _copy_run_files(source, temp / "legacy")
        if manifest(temp / "legacy") != before or _run_manifest(source) != before:
            raise ValueError(f"Source run changed during copy: {source}")
        write_json(temp / "migration.json", data)
        if "config" in data:
            write_json(temp / "config.json", data["config"])
        for name in ("summary.json", "samples.jsonl"):
            if name in before:
                shutil.copy2(temp / "legacy" / name, temp / name)
                if file_hash(temp / name) != before[name]:
                    raise ValueError(f"Run artifact copy failed verification: {name}")
        rename_new(temp, destination)
        return "migrated"
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def publish_run_link(experiments: Path, run_id: str, run_path: Path) -> None:
    if not run_id or not run_id.isalnum():
        raise ValueError("Invalid MLflow run ID")
    directory = experiments / "intervention_continuation" / "migration_runs"
    destination = directory / f"{run_id}.json"
    require_safe_path(destination, experiments)
    data = {"run_id": run_id, "run_path": run_path.relative_to(experiments).as_posix()}
    publish_json(destination, data)
