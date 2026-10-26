"""Simple, file-based execution storage utilities and FileStorage service."""

import json
import uuid
from pathlib import Path
from typing import Any

import torch

from ...domain.models.tensor_artifact import (
    ImmutableTensorOverwriteError,
    TensorMissingError,
)
from ..paths.resolver import PathResolver, get_path_resolver


def save_tensor(path: Path | str, tensor: Any, overwrite: bool = False) -> Path:
    """Atomically save a torch tensor or KV cache structure via a unique temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and not overwrite:
        raise ImmutableTensorOverwriteError(
            f"Cannot overwrite immutable tensor at '{target}' without overwrite=True."
        )

    tmp = target.with_name(f"{target.name}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        torch.save(tensor, tmp)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def load_tensor(path: Path | str, map_location: Any = None) -> Any:
    """Load a torch tensor or KV cache structure from disk."""
    target = Path(path)
    if not target.is_file():
        raise TensorMissingError(f"Tensor file not found on disk: {target}")
    return torch.load(target, map_location=map_location, weights_only=False)


def save_json_atomic(path: Path | str, data: Any) -> Path:
    """Atomically write data as JSON using a unique temp file and atomic replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{uuid.uuid4().hex[:8]}")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def load_json(path: Path | str) -> Any:
    """Load and parse JSON from disk."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"JSON file not found: {target}")
    with open(target, encoding="utf-8") as f:
        return json.load(f)


def is_condition_complete(condition_dir: Path | str) -> bool:
    """Judge condition completion based on valid result.json, no errors, and required files."""
    cdir = Path(condition_dir)
    res_path = cdir / "result.json"
    if not res_path.is_file():
        return False
    try:
        data = load_json(res_path)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False

    if data.get("error") is not None or data.get("error_msg") is not None:
        return False

    required_files = data.get("files", [])
    return all((cdir / fname).is_file() for fname in required_files)


def is_experiment_complete(
    exp_dir: Path | str,
    target_count: int = 0,
    conditions: list[str] | None = None,
) -> bool:
    """Judge experiment completion: verifies that all required sample conditions are complete on disk."""
    edir = Path(exp_dir)
    conds = conditions or ["benchmark"]
    resolver = get_path_resolver()

    if target_count <= 0:
        samples_dir = edir / "samples"
        if not samples_dir.is_dir():
            return False
        sample_subdirs = [d for d in samples_dir.iterdir() if d.is_dir()]
        target_count = len(sample_subdirs)
        if target_count <= 0:
            return False

    for idx in range(target_count):
        for cond in conds:
            cdir = resolver.get_condition_dir(edir, idx, cond)
            if not is_condition_complete(cdir):
                if cond == "benchmark" and is_condition_complete(
                    edir / "samples" / f"{idx:06d}"
                ):
                    continue
                return False
    return True


def save_condition_result(
    condition_dir: Path | str,
    result_data: dict[str, Any],
    tensors: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Save condition tensors first, then atomically commit result.json."""
    cdir = Path(condition_dir)
    cdir.mkdir(parents=True, exist_ok=True)

    if not overwrite and is_condition_complete(cdir):
        return cdir / "result.json"

    if result_data.get("error") is not None or result_data.get("error_msg") is not None:
        err_path = cdir / "error.json"
        save_json_atomic(err_path, result_data)
        return err_path

    effective_overwrite = overwrite or (not is_condition_complete(cdir))

    file_list = list(result_data.get("files", []))
    if tensors:
        for name, t in tensors.items():
            fname = name if name.endswith(".pt") else f"{name}.pt"
            save_tensor(cdir / fname, t, overwrite=effective_overwrite)
            if fname not in file_list:
                file_list.append(fname)

    payload = dict(result_data)
    payload["files"] = file_list
    return save_json_atomic(cdir / "result.json", payload)





class FileStorage:
    """File-based experiment storage rooted at PathResolver.experiments_dir."""

    def __init__(self, path_resolver: PathResolver | None = None) -> None:
        self._resolver = path_resolver or get_path_resolver()

    @property
    def base_dir(self) -> Path:
        return self._resolver.experiments_dir

    def get_run_dir(self, run_id: str) -> Path:
        return self._resolver.get_experiment_dir(run_id)

    def get_sample_dir(self, run_id: str, sample_key: str) -> Path:
        return self._resolver.get_sample_dir(run_id, sample_key)

    def get_condition_dir(self, run_id: str, sample_key: str, condition: str) -> Path:
        return self._resolver.get_condition_dir(run_id, sample_key, condition)

    def is_condition_complete(
        self, run_id: str, sample_key: str, condition: str
    ) -> bool:
        cdir = self.get_condition_dir(run_id, sample_key, condition)
        return is_condition_complete(cdir)

    def save_condition(
        self,
        run_id: str,
        sample_key: str,
        condition: str,
        result_data: dict[str, Any],
        tensors: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        cdir = self.get_condition_dir(run_id, sample_key, condition)
        return save_condition_result(
            cdir, result_data, tensors=tensors, overwrite=overwrite
        )

    def load_condition_result(
        self, run_id: str, sample_key: str, condition: str
    ) -> dict[str, Any]:
        cdir = self.get_condition_dir(run_id, sample_key, condition)
        return load_json(cdir / "result.json")

    def load_condition_tensor(
        self,
        run_id: str,
        sample_key: str,
        condition: str,
        name: str = "hidden",
        map_location: Any = None,
    ) -> Any:
        cdir = self.get_condition_dir(run_id, sample_key, condition)
        fname = name if name.endswith(".pt") else f"{name}.pt"
        return load_tensor(cdir / fname, map_location=map_location)

    def save_tensor(
        self,
        path: Path | str,
        tensor: Any,
        overwrite: bool = False,
    ) -> Path:
        return save_tensor(path, tensor, overwrite=overwrite)

    def load_tensor(
        self,
        path: Path | str,
        map_location: Any = None,
    ) -> Any:
        return load_tensor(path, map_location=map_location)


DEFAULT_FILE_STORAGE = FileStorage()


def get_file_storage() -> FileStorage:
    return DEFAULT_FILE_STORAGE
