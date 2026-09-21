import os
import secrets
import subprocess
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType
from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from ...domain.ports.tracking_port import ExperimentTrackerPort, LiveSpanPort
from ..paths.resolver import PathResolver, get_path_resolver

with suppress(Exception):
    from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

    RandomIdGenerator.generate_trace_id = lambda self: secrets.randbits(128)
    RandomIdGenerator.generate_span_id = lambda self: secrets.randbits(64)


class MLflowSpanAdapter(LiveSpanPort):
    """Adapter wrapping MLflow LiveSpan to fulfill LiveSpanPort."""

    def __init__(self, span: Any) -> None:
        self._span = span

    def set_inputs(self, inputs: dict[str, Any]) -> None:
        if self._span is not None:
            self._span.set_inputs(inputs)

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        if self._span is not None:
            self._span.set_outputs(outputs)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self._span is not None:
            usage = {
                TokenUsageKey.INPUT_TOKENS: int(prompt_tokens),
                TokenUsageKey.OUTPUT_TOKENS: int(completion_tokens),
                TokenUsageKey.TOTAL_TOKENS: int(prompt_tokens + completion_tokens),
            }
            self._span.set_attribute(SpanAttributeKey.CHAT_USAGE, usage)

    def set_status(self, status: str, description: str | None = None) -> None:
        if self._span is not None:
            self._span.set_status(status, description=description)

    @property
    def trace_id(self) -> str | None:
        return getattr(self._span, "trace_id", None)


def get_git_commit_hash(repo_dir: Path | None = None) -> str:
    """Safely obtain current git commit hash, or return 'unknown'."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        cwd = str(repo_dir) if repo_dir else None
        res = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "unknown"


class MLflowTracker(ExperimentTrackerPort):
    """MLflow experiment tracker adapter.

    Features:
    - Automatically defaults to local SQLite at .cache/mlflow.db and local artifact
      directory at .cache/mlflow_artifacts if MLFLOW_TRACKING_URI is not set.
    - Automatically delegates to remote tracking server and server-side artifact
      configuration when MLFLOW_TRACKING_URI is supplied.
    """

    def __init__(
        self,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
        path_resolver: PathResolver | None = None,
    ) -> None:
        self.resolver = path_resolver or get_path_resolver()
        env_uri = os.environ.get("MLFLOW_TRACKING_URI")

        if tracking_uri is not None:
            self.tracking_uri = tracking_uri
            self.is_remote = not tracking_uri.startswith(
                "sqlite"
            ) and not tracking_uri.startswith("file")
        elif env_uri is not None and env_uri.strip():
            self.tracking_uri = env_uri.strip()
            self.is_remote = not self.tracking_uri.startswith(
                "sqlite"
            ) and not self.tracking_uri.startswith("file")
        else:
            # Default to local SQLite
            db_path = self.resolver.cache_dir / "mlflow.db"
            self.tracking_uri = f"sqlite:///{db_path.resolve()}"
            self.is_remote = False

        mlflow.set_tracking_uri(self.tracking_uri)

        if not self.is_remote:
            self.artifact_location = artifact_location or str(
                (self.resolver.cache_dir / "mlflow_artifacts").resolve()
            )
            Path(self.artifact_location).mkdir(parents=True, exist_ok=True)
        else:
            # When remote, follow remote server-side artifact configuration without forcing local path
            self.artifact_location = None

        self._active_run = None

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        run_tags = dict(tags) if tags else {}
        run_tags.setdefault("git_commit", get_git_commit_hash(self.resolver.root))

        # Check existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if (
            experiment is not None
            and getattr(experiment, "lifecycle_stage", "") == "deleted"
        ):
            with suppress(Exception):
                mlflow.tracking.MlflowClient().restore_experiment(
                    experiment.experiment_id
                )
            experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            if not self.is_remote and self.artifact_location:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=self.artifact_location,
                )
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        self._active_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name,
            tags=run_tags,
        )
        return self._active_run

    @property
    def active_run_id(self) -> str | None:
        if self._active_run is not None:
            return getattr(self._active_run.info, "run_id", None)
        active = mlflow.active_run()
        return active.info.run_id if active is not None else None

    def log_params(self, params: dict[str, Any]) -> None:
        flat_params: dict[str, Any] = {}
        for k, v in params.items():
            if v is None:
                flat_params[k] = "None"
            elif isinstance(v, (list, tuple, dict)):
                flat_params[k] = str(v)
            else:
                flat_params[k] = v
        # MLflow parameter values truncated to 500 characters
        safe_params = {
            k: (str(v)[:500] if len(str(v)) > 500 else v)
            for k, v in flat_params.items()
        }
        mlflow.log_params(safe_params)

    def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ) -> None:
        clean_metrics = {k: float(v) for k, v in metrics.items()}
        mlflow.log_metrics(clean_metrics, step=step)

    def log_artifact(
        self, local_path: Path | str, artifact_path: str | None = None
    ) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(dictionary, artifact_file)

    def end_run(self, status: str = "FINISHED") -> None:
        if mlflow.active_run() is not None:
            mlflow.end_run(status=status)
        self._active_run = None

    @contextmanager
    def start_sample_trace(
        self,
        name: str,
        inputs: dict[str, Any],
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        """Start a root sample trace."""
        with mlflow.start_span(name=name, span_type=SpanType.AGENT) as span:
            span.set_inputs(inputs)
            if tags or request_preview:
                safe_tags = {str(k): str(v) for k, v in tags.items()} if tags else None
                mlflow.update_current_trace(
                    tags=safe_tags, request_preview=request_preview
                )
            adapter = MLflowSpanAdapter(span)
            yield adapter

    @contextmanager
    def start_span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        inputs: dict[str, Any] | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        """Start a child span within the active trace."""
        st = getattr(SpanType, span_type.upper(), SpanType.UNKNOWN)
        with mlflow.start_span(name=name, span_type=st) as span:
            if inputs is not None:
                span.set_inputs(inputs)
            adapter = MLflowSpanAdapter(span)
            yield adapter

    def update_current_trace(
        self,
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        """Update active trace tags and previews."""
        safe_tags = {str(k): str(v) for k, v in tags.items()} if tags else None
        mlflow.update_current_trace(
            tags=safe_tags,
            request_preview=request_preview,
            response_preview=response_preview,
        )

    def log_expectation(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "ground_truth",
    ) -> None:
        """Log expectation (ground truth) assessment on a trace."""
        try:
            self.flush_traces()
            source = AssessmentSource(
                source_type=AssessmentSourceType.CODE, source_id=source_id
            )
            mlflow.log_expectation(
                trace_id=trace_id,
                name=name,
                value=value,
                source=source,
            )
        except Exception:
            pass

    def log_feedback(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "evaluator",
        rationale: str | None = None,
    ) -> None:
        """Log feedback assessment on a trace."""
        try:
            self.flush_traces()
            source = AssessmentSource(
                source_type=AssessmentSourceType.CODE, source_id=source_id
            )
            mlflow.log_feedback(
                trace_id=trace_id,
                name=name,
                value=value,
                source=source,
                rationale=rationale,
            )
        except Exception:
            pass

    def flush_traces(self) -> None:
        """Flush background async trace queue."""
        mlflow.flush_trace_async_logging()
