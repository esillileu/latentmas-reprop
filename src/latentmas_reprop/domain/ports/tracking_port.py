from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ExperimentTrackerPort(ABC):
    """Port interface for experiment tracking (MLflow, etc.)."""

    @abstractmethod
    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        """Start a new experiment tracking run."""
        raise NotImplementedError

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.log_params({key: value})

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameter dictionary."""
        raise NotImplementedError

    def log_metric(
        self, key: str, value: float | int, step: int | None = None
    ) -> None:
        """Log a single evaluation metric."""
        self.log_metrics({key: value}, step=step)

    @abstractmethod
    def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ) -> None:
        """Log evaluation metric dictionary."""
        raise NotImplementedError

    @abstractmethod
    def log_artifact(
        self, local_path: Path | str, artifact_path: str | None = None
    ) -> None:
        """Log a local file or directory as an artifact."""
        raise NotImplementedError

    @abstractmethod
    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Log a dictionary directly as a JSON/YAML artifact."""
        raise NotImplementedError

    @abstractmethod
    def end_run(self) -> None:
        """End the current tracking run."""
        raise NotImplementedError
