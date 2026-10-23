import json
import shutil
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

    def clean_volatile_cache(
        self,
        include_resumable_runs: bool = False,
        purge_incomplete_runs: bool = False,
    ) -> list[Path]:
        """Safely clean volatile execution caches without touching MLflow or storage.

        To prevent data loss from un-uploaded or in-progress experiments,
        evaluation runs (CacheLayer.EVALUATION_RUNS containing BenchmarkRunStore
        checkpoints) are preserved by default. When include_resumable_runs=True,
        only completed runs (with summary.json) are removed unless
        purge_incomplete_runs=True is explicitly given.
        """
        import logging

        logger = logging.getLogger(__name__)

        cleaned = []
        # Pure scratch layers that can be safely wiped at any time
        volatile_layers = [
            CacheLayer.MODELS_REALIGN,
            CacheLayer.EVALUATION_INTERVENTIONS,
            CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
            CacheLayer.LATENT_RECEIVER_ACQUISITION,
        ]
        for layer in volatile_layers:
            layer_dir = self.get_layer_path(layer)
            if layer_dir.exists() and self._resolver.is_safe_cache_path(layer_dir):
                shutil.rmtree(layer_dir, ignore_errors=True)
                cleaned.append(layer_dir)

        # Handle runtime scratch with protection for active in-progress checkpoints
        runtime_dir = self.get_layer_path(CacheLayer.RUNTIME)
        if runtime_dir.exists() and self._resolver.is_safe_cache_path(runtime_dir):
            for item in list(runtime_dir.iterdir()):
                is_checkpoint = "checkpoint" in item.name and item.suffix == ".jsonl"
                if is_checkpoint and not purge_incomplete_runs:
                    logger.info("Preserving active runtime checkpoint: %s", item)
                    continue
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
                cleaned.append(item)

        # Handle evaluation runs with granular protection for incomplete checkpoints
        runs_dir = self.get_layer_path(CacheLayer.EVALUATION_RUNS)
        if runs_dir.exists() and self._resolver.is_safe_cache_path(runs_dir):
            if not include_resumable_runs:
                logger.info(
                    "Preserving evaluation runs at %s (use include_resumable_runs=True to clean)",
                    runs_dir,
                )
            else:
                for run_path in list(runs_dir.iterdir()):
                    if not run_path.is_dir():
                        continue
                    samples_file = run_path / "samples.jsonl"
                    summary_file = run_path / "summary.json"
                    is_incomplete = (
                        samples_file.is_file() and not summary_file.is_file()
                    )

                    if is_incomplete and not purge_incomplete_runs:
                        logger.warning(
                            "Preserved incomplete evaluation run checkpoint: %s",
                            run_path,
                        )
                        continue

                    shutil.rmtree(run_path, ignore_errors=True)
                    cleaned.append(run_path)

                # Remove empty runs_dir if empty
                if not any(runs_dir.iterdir()):
                    shutil.rmtree(runs_dir, ignore_errors=True)
                    cleaned.append(runs_dir)

        return cleaned


DEFAULT_CACHE_MANAGER = ExecutionCacheManager()


def get_cache_manager() -> ExecutionCacheManager:
    return DEFAULT_CACHE_MANAGER
