"""Unified ResearchDataReader for loading experiment artifacts and tensors."""

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from ..paths.resolver import PathResolver, get_path_resolver
from .tensor_store import (
    FileStorage,
    get_file_storage,
    load_json,
    load_tensor,
)


def _validate_safe_name(name_val: str, field_name: str) -> str:
    """Disallow path traversal components and directory separators for leaf names."""
    if not name_val:
        return name_val
    if ".." in name_val or "/" in name_val or "\\" in name_val:
        raise ValueError(
            f"Path traversal or invalid path character in {field_name}: '{name_val}'"
        )
    return name_val


def _validate_safe_path(path_val: str, field_name: str) -> str:
    """Disallow path traversal components and absolute paths for hierarchical paths."""
    if not path_val:
        return path_val
    p = Path(path_val)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(
            f"Path traversal or absolute path disallowed in {field_name}: '{path_val}'"
        )
    return path_val


def _validate_safe_relpath(path_val: str, field_name: str) -> Path:
    """Disallow path traversal components and absolute paths for relative subpaths."""
    p = Path(path_val)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(
            f"Path traversal or absolute path disallowed in {field_name}: '{path_val}'"
        )
    return p


class ResearchDataReader:
    """Unified reader interface for experiment artifacts and tensors.

    Abstracts away local filesystem layouts and MLflow artifact retrieval,
    allowing downstream analysis tools to query data by Run ID and Artifact ID
    without hardcoding paths.
    """

    def __init__(
        self,
        mlflow_client: Any | None = None,
        tensor_store: FileStorage | None = None,
        path_resolver: PathResolver | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.client = mlflow_client
        self.resolver = path_resolver or get_path_resolver()
        self.tensor_store = tensor_store or get_file_storage()
        self.cache_dir = cache_dir or (self.resolver.cache_dir / "reader_scratch")
        self.provenance: list[dict[str, Any]] = []

    def get_run_data_dir(self, run_id: str) -> Path:
        """Find the local storage directory associated with an experiment run or relative path without mutating disk."""
        _validate_safe_path(run_id, "run_id")

        # 1. Check MLflow run tags if client is available
        if self.client:
            with contextlib.suppress(Exception):
                run = self.client.get_run(run_id)
                tag_dir = run.data.tags.get("data_dir")
                if tag_dir:
                    return Path(tag_dir)
                alt_id = run.data.tags.get("rel_dir") or run.data.tags.get(
                    "resumed_from_run_id"
                )
                if alt_id:
                    _validate_safe_path(alt_id, "rel_dir")
                    return self.resolver.get_experiment_dir(alt_id)

            with contextlib.suppress(Exception):
                for exp in self.client.search_experiments():
                    runs = self.client.search_runs(
                        [exp.experiment_id],
                        filter_string=f"tags.rel_dir = '{run_id}'",
                        max_results=1,
                    )
                    if runs:
                        tag_dir = runs[0].data.tags.get("data_dir")
                        if tag_dir:
                            return Path(tag_dir)

        clean_rel = str(run_id).strip("/\\")
        return self.resolver.experiments_dir / clean_rel

    def _intervention_state_dir(
        self, run_or_state: str
    ) -> tuple[Path, dict[str, Any] | None]:
        _validate_safe_path(run_or_state, "run_or_state")
        link = (
            self.resolver.experiments_dir
            / "intervention_continuation"
            / "migration_runs"
            / f"{run_or_state}.json"
        )
        if "/" not in run_or_state and link.is_file():
            reference = load_json(link)
            if reference.get("run_id") != run_or_state:
                raise ValueError("Invalid intervention migration run link")
            relative = _validate_safe_relpath(reference["run_path"], "run_path")
            path = self.resolver.experiments_dir / relative
        else:
            path = self.get_run_data_dir(run_or_state)
        config_path = path / "config.json"
        if config_path.is_file():
            config = load_json(config_path)
            relative = _validate_safe_relpath(config["state_path"], "state_path")
            return self.resolver.experiments_dir / relative, config
        return path, None

    def load_intervention_state(
        self, run_or_state: str, sample: str, step: int | None = None
    ) -> tuple[Any, torch.Tensor, dict[str, Any]]:
        _validate_safe_name(sample, "sample")
        state, config = self._intervention_state_dir(run_or_state)
        if step is None:
            if config is None:
                raise ValueError("A step is required for a state path")
            step = int(config["step"])
        return self.tensor_store.assemble_shared_state(state, sample, step)

    def load_intervention_result(
        self,
        run_or_state: str,
        sample: str,
        step: int,
        result_hash: str,
        condition: str,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        for value, name in (
            (sample, "sample"),
            (result_hash, "result_hash"),
            (condition, "condition"),
        ):
            _validate_safe_name(value, name)
        state, _ = self._intervention_state_dir(run_or_state)
        directory = (
            state
            / "samples"
            / sample
            / "results"
            / f"ls{step}"
            / result_hash
            / condition
        )
        return self.tensor_store.load_intervention_result(directory)

    def load_artifact(self, run_id: str, artifact_path: str) -> Any:
        """Load an artifact (JSON, JSONL, YAML) from MLflow or local run storage."""
        _validate_safe_path(run_id, "run_id")
        rel_path = _validate_safe_relpath(artifact_path, "artifact_path")
        downloaded_path = None

        run_dir = self.get_run_data_dir(run_id)
        candidate = run_dir / rel_path
        if candidate.is_file():
            downloaded_path = candidate

        if (downloaded_path is None or not downloaded_path.is_file()) and self.client:
            actual_run_id = run_id
            if "/" in run_id:
                with contextlib.suppress(Exception):
                    for exp in self.client.search_experiments():
                        runs = self.client.search_runs(
                            [exp.experiment_id],
                            filter_string=f"tags.rel_dir = '{run_id}'",
                            max_results=1,
                        )
                        if runs:
                            actual_run_id = runs[0].info.run_id
                            break
            dest_dir = self.cache_dir / actual_run_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                downloaded_path = Path(
                    self.client.download_artifacts(
                        actual_run_id, artifact_path, str(dest_dir)
                    )
                )

        if downloaded_path is None or not downloaded_path.is_file():
            raise FileNotFoundError(
                f"Artifact '{artifact_path}' not found for run '{run_id}'."
            )

        self.provenance.append(
            {
                "run_id": run_id,
                "artifact_path": artifact_path,
                "local_path": str(downloaded_path),
            }
        )

        if artifact_path.endswith(".jsonl"):
            return [
                json.loads(line)
                for line in downloaded_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif artifact_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(downloaded_path.read_text(encoding="utf-8"))
        elif artifact_path.endswith(".json"):
            return load_json(downloaded_path)
        return downloaded_path.read_text(encoding="utf-8")

    def load_tensor(
        self,
        run_id: str,
        artifact_id: str,
        verify_checksum: bool = False,
        map_location: Any = None,
        sample_key: str = "",
        condition: str = "",
        branch_id: str = "",
        segment_id: str = "",
    ) -> torch.Tensor:
        """Load a torch tensor for a given run and artifact identifier."""
        _validate_safe_path(run_id, "run_id")
        run_dir = self.get_run_data_dir(run_id)
        cond = condition or branch_id
        fname = artifact_id if artifact_id.endswith(".pt") else f"{artifact_id}.pt"

        # 1. Exact sample and condition lookup
        if sample_key and cond:
            _validate_safe_name(sample_key, "sample_key")
            _validate_safe_name(cond, "condition")
            _validate_safe_name(fname, "artifact_id")
            candidate = run_dir / "samples" / sample_key / cond / fname
            if candidate.is_file():
                return load_tensor(candidate, map_location=map_location)
            raise FileNotFoundError(
                f"Tensor artifact '{fname}' not found for sample '{sample_key}' condition '{cond}' in run '{run_id}'."
            )

        # 2. Shared context lookup for sample (condition not specified)
        if sample_key:
            _validate_safe_name(sample_key, "sample_key")
            _validate_safe_name(fname, "artifact_id")
            candidate = run_dir / "samples" / sample_key / "shared" / fname
            if candidate.is_file():
                return load_tensor(candidate, map_location=map_location)
            raise FileNotFoundError(
                f"Shared tensor artifact '{fname}' not found for sample '{sample_key}' in run '{run_id}'."
            )

        # 3. Direct run-level tensor check
        if "/" not in artifact_id and "\\" not in artifact_id:
            _validate_safe_name(fname, "artifact_id")
            candidate = run_dir / fname
            if candidate.is_file():
                return load_tensor(candidate, map_location=map_location)

        # 4. Explicit relative subpath in run_dir
        rel_subpath = _validate_safe_relpath(artifact_id, "artifact_id")
        candidate = run_dir / rel_subpath
        if candidate.is_file():
            return load_tensor(candidate, map_location=map_location)

        raise FileNotFoundError(
            f"Tensor artifact '{artifact_id}' not found for run '{run_id}'."
        )
