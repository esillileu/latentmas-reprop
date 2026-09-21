from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class LiveSpanPort(ABC):
    """Port interface for an active tracing span."""

    @abstractmethod
    def set_inputs(self, inputs: dict[str, Any]) -> None:
        """Record span inputs."""
        raise NotImplementedError

    @abstractmethod
    def set_outputs(self, outputs: dict[str, Any]) -> None:
        """Record span outputs."""
        raise NotImplementedError

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single metadata attribute on span."""
        raise NotImplementedError

    @abstractmethod
    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Record standardized token usage on the span."""
        raise NotImplementedError

    @abstractmethod
    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span status ('OK' or 'ERROR') with optional error description."""
        raise NotImplementedError

    @property
    @abstractmethod
    def trace_id(self) -> str | None:
        """Return the trace ID associated with this span."""
        raise NotImplementedError


class DummySpan(LiveSpanPort):
    """Safe no-op span implementation for non-tracing trackers."""

    def set_inputs(self, inputs: dict[str, Any]) -> None:
        pass

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        pass

    def set_status(self, status: str, description: str | None = None) -> None:
        pass

    @property
    def trace_id(self) -> str | None:
        return None


class ExperimentTrackerPort(ABC):
    """Port interface for experiment tracking and GenAI tracing (MLflow, etc.)."""

    @abstractmethod
    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        """Start a new experiment tracking run."""
        raise NotImplementedError

    @property
    def active_run_id(self) -> str | None:
        """Return currently active run ID if any."""
        return None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self.log_params({key: value})

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameter dictionary."""
        raise NotImplementedError

    def log_metric(self, key: str, value: float | int, step: int | None = None) -> None:
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
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current tracking run with the specified status."""
        raise NotImplementedError

    @contextmanager
    def start_sample_trace(
        self,
        name: str,
        inputs: dict[str, Any],
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        """Start a root sample trace."""
        yield DummySpan()

    @contextmanager
    def start_span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        inputs: dict[str, Any] | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        """Start a child span."""
        yield DummySpan()

    def update_current_trace(
        self,
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        """Update trace metadata, previews, and tags."""
        return None  # Default no-op for non-tracing trackers

    def log_expectation(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "ground_truth",
    ) -> None:
        """Log ground truth expectation assessment."""
        return None  # Default no-op for non-tracing trackers

    def log_feedback(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "evaluator",
        rationale: str | None = None,
    ) -> None:
        """Log feedback assessment."""
        return None  # Default no-op for non-tracing trackers

    def flush_traces(self) -> None:
        """Flush any pending async traces."""
        return None  # Default no-op for non-tracing trackers
