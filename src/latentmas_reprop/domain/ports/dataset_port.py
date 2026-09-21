from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any


class DatasetPort(ABC):
    """Port interface for loading benchmark datasets."""

    @abstractmethod
    def load(
        self,
        task: str,
        split: str = "test",
        **kwargs: Any,
    ) -> Iterable[dict]:
        """Load items for the given benchmark task and split."""
        raise NotImplementedError
