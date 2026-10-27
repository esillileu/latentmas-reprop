import copy
from pathlib import Path
from typing import Any

import torch

from ...domain.models.tensor_artifact import (
    ImmutableTensorOverwriteError,
    TensorMissingError,
)


class InMemoryStorage:
    """Fast in-memory condition and tensor storage for unit tests without disk I/O."""

    def __init__(self, base_dir: Path | str = "/tmp/in_memory_storage") -> None:
        self.base_dir = Path(base_dir)
        self._conditions: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._tensors: dict[tuple[str, str, str, str], Any] = {}
        self._raw_tensors: dict[str, Any] = {}

    def get_run_dir(self, run_id: str) -> Path:
        return self.base_dir / run_id

    def get_sample_dir(self, run_id: str, sample_key: str) -> Path:
        return self.get_run_dir(run_id) / "samples" / str(sample_key)

    def get_condition_dir(self, run_id: str, sample_key: str, condition: str) -> Path:
        return self.get_sample_dir(run_id, sample_key) / condition

    def is_condition_complete(self, run_id: str, sample_key: str, condition: str) -> bool:
        key = (str(run_id), str(sample_key), str(condition))
        if key not in self._conditions:
            return False
        data = self._conditions[key]
        if data.get("error") is not None or data.get("error_msg") is not None:
            return False
        required = data.get("files", [])
        return all((*key, fname.removesuffix(".pt")) in self._tensors for fname in required if fname.endswith(".pt"))

    def save_condition(
        self,
        run_id: str,
        sample_key: str,
        condition: str,
        result_data: dict[str, Any],
        tensors: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Path:
        key = (str(run_id), str(sample_key), str(condition))
        if not overwrite and self.is_condition_complete(run_id, sample_key, condition):
            return self.get_condition_dir(run_id, sample_key, condition) / "result.json"

        file_list = list(result_data.get("files", []))
        if tensors:
            for name, t in tensors.items():
                clean_name = name.removesuffix(".pt")
                t_key = (*key, clean_name)
                if t_key in self._tensors and not overwrite:
                    raise ImmutableTensorOverwriteError(f"Tensor {clean_name} already exists.")
                self._tensors[t_key] = t.clone() if isinstance(t, torch.Tensor) else copy.deepcopy(t)
                fname = f"{clean_name}.pt"
                if fname not in file_list:
                    file_list.append(fname)

        payload = copy.deepcopy(result_data)
        payload["files"] = file_list
        self._conditions[key] = payload
        return self.get_condition_dir(run_id, sample_key, condition) / "result.json"

    def load_condition_result(self, run_id: str, sample_key: str, condition: str) -> dict[str, Any]:
        key = (str(run_id), str(sample_key), str(condition))
        if key not in self._conditions:
            raise FileNotFoundError(f"Condition not found: {key}")
        return copy.deepcopy(self._conditions[key])

    def load_condition_tensor(
        self,
        run_id: str,
        sample_key: str,
        condition: str,
        name: str = "hidden",
        map_location: Any = None,
    ) -> Any:
        clean_name = name.removesuffix(".pt")
        key = (str(run_id), str(sample_key), str(condition), clean_name)
        if key not in self._tensors:
            raise TensorMissingError(f"Tensor {clean_name} not found in {key[:3]}")
        t = self._tensors[key]
        if isinstance(t, torch.Tensor):
            return t.to(map_location) if map_location else t.clone()
        return copy.deepcopy(t)

    def save_tensor(self, path: Path | str, tensor: Any, overwrite: bool = False) -> Path:
        p_str = str(path)
        if p_str in self._raw_tensors and not overwrite:
            raise ImmutableTensorOverwriteError(f"Raw tensor at {p_str} already exists.")
        self._raw_tensors[p_str] = tensor.clone() if isinstance(tensor, torch.Tensor) else copy.deepcopy(tensor)
        return Path(path)

    def load_tensor(self, path: Path | str, map_location: Any = None) -> Any:
        p_str = str(path)
        if p_str not in self._raw_tensors:
            raise TensorMissingError(f"Raw tensor not found: {p_str}")
        t = self._raw_tensors[p_str]
        if isinstance(t, torch.Tensor):
            return t.to(map_location) if map_location else t.clone()
        return copy.deepcopy(t)
