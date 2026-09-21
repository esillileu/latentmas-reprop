import json
from pathlib import Path
from typing import Any

import torch

from ...domain.ports.cache_port import CacheLayer, CachePort
from ..paths.resolver import PathResolver, get_path_resolver


class ExecutionCacheManager(CachePort):
    """Hierarchical execution cache manager rooted at repository's .cache directory.

    Organizes non-HF caches by usage layer:
    - models/realign: realignment matrices
    - evaluation/runs: benchmark prediction logs & results
    - datasets: non-HF/preprocessed dataset caches
    - runtime: intermediate execution scratch
    """

    def __init__(self, path_resolver: PathResolver | None = None) -> None:
        self._resolver = path_resolver or get_path_resolver()

    @property
    def base_dir(self) -> Path:
        return self._resolver.cache_dir

    def _resolve_layer_name(self, layer: CacheLayer | str) -> str:
        return layer.value if isinstance(layer, CacheLayer) else str(layer)

    def get_layer_path(self, layer: CacheLayer | str) -> Path:
        layer_str = self._resolve_layer_name(layer)
        return self._resolver.get_cache_layer_dir(layer_str)

    def _get_file_path(self, layer: CacheLayer | str, filename: str) -> Path:
        layer_dir = self.get_layer_path(layer)
        return layer_dir / filename

    def exists(self, layer: CacheLayer | str, filename: str) -> bool:
        return self._get_file_path(layer, filename).exists()

    def save_json(
        self, layer: CacheLayer | str, filename: str, data: Any, indent: int = 2
    ) -> Path:
        path = self._get_file_path(layer, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        return path

    def load_json(self, layer: CacheLayer | str, filename: str) -> Any:
        path = self._get_file_path(layer, filename)
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def save_torch(self, layer: CacheLayer | str, filename: str, obj: Any) -> Path:
        path = self._get_file_path(layer, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)
        return path

    def load_torch(
        self, layer: CacheLayer | str, filename: str, map_location: Any = None
    ) -> Any:
        path = self._get_file_path(layer, filename)
        return torch.load(path, map_location=map_location)


DEFAULT_CACHE_MANAGER = ExecutionCacheManager()


def get_cache_manager() -> ExecutionCacheManager:
    return DEFAULT_CACHE_MANAGER
