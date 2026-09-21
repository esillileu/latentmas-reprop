from collections.abc import Callable, Iterable
from typing import Any

from ...domain.ports.dataset_port import DatasetPort
from .loaders import (
    load_aime2024,
    load_aime2025,
    load_arc_challenge,
    load_arc_easy,
    load_gpqa_diamond,
    load_gsm8k,
    load_humanevalplus,
    load_mbppplus,
    load_medqa,
    load_winogrande,
)


class DatasetRegistry(DatasetPort):
    """Registry and loader for all benchmark datasets."""

    def __init__(self) -> None:
        self._loaders: dict[str, Callable[..., Iterable[dict]]] = {
            "gsm8k": load_gsm8k,
            "aime2024": lambda split="train", **kw: load_aime2024(split="train", **kw),
            "aime2025": lambda split="train", **kw: load_aime2025(split="train", **kw),
            "gpqa": load_gpqa_diamond,
            "arc_easy": load_arc_easy,
            "arc_challenge": load_arc_challenge,
            "winogrande": load_winogrande,
            "mbppplus": load_mbppplus,
            "humanevalplus": load_humanevalplus,
            "medqa": load_medqa,
        }

    def load(self, task: str, split: str = "test", **kwargs: Any) -> Iterable[dict]:
        task_lower = task.lower()
        if task_lower not in self._loaders:
            supported = list(self._loaders.keys())
            raise ValueError(f"No {task} support. Supported tasks: {supported}")
        loader = self._loaders[task_lower]
        return loader(split=split, **kwargs)


DEFAULT_DATASET_REGISTRY = DatasetRegistry()


def get_dataset_registry() -> DatasetRegistry:
    return DEFAULT_DATASET_REGISTRY
