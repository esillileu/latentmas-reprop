import os
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import SpanType
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType

from ...domain.ports.tracking_port import ExperimentTrackerPort, LiveSpanPort
from ..paths.resolver import PathResolver, get_git_commit_hash, get_path_resolver
from .span_adapter import MLflowSpanAdapter


def _truncate_preview(text: str | None, max_length: int = 1000) -> str | None:
    if text is None or len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


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
        elif env_uri is not None and env_uri.strip():
            self.tracking_uri = env_uri.strip()
        else:
            # Standard persistent storage MLflow (storage/mlflow/mlflow.db)
            db_path = self.resolver.mlflow_db_path
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.tracking_uri = f"sqlite:///{db_path.resolve()}"
        self.is_remote = not self.tracking_uri.startswith(("sqlite", "file"))

        mlflow.set_tracking_uri(self.tracking_uri)

        if not self.is_remote:
            self.artifact_location = artifact_location or str(
                self.resolver.mlflow_artifacts_dir.resolve()
            )
            Path(self.artifact_location).mkdir(parents=True, exist_ok=True)
        else:
            # When remote, follow remote server-side artifact configuration without forcing local path
            self.artifact_location = None

        self._active_run = None
        self._assessment_ready_traces = set()

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        self._assessment_ready_traces.clear()
        run_tags = {str(key): str(value) for key, value in (tags or {}).items()}
        run_tags.setdefault("git_commit", get_git_commit_hash(self.resolver.root))

        # Check existing experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None and getattr(experiment, "lifecycle_stage", "") == "deleted":
            with suppress(Exception):
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
            experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            loc = self.artifact_location if not self.is_remote else None
            experiment_id = mlflow.create_experiment(experiment_name, artifact_location=loc) if loc else mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id

        # Canonical experiment path and directory tags
        rel_dir = run_tags.get("rel_dir") or (str(run_name) if run_name and ("/" in str(run_name) or run_tags.get("task")) else None)
        data_dir = run_tags.get("data_dir") or (str(self.resolver.get_experiment_dir(rel_dir).resolve()) if rel_dir else None)
        if run_name and "run_name" not in run_tags:
            run_tags["run_name"] = str(run_name)
        if rel_dir and "rel_dir" not in run_tags:
            run_tags["rel_dir"] = str(rel_dir)
        if data_dir and "data_dir" not in run_tags:
            run_tags["data_dir"] = str(data_dir)

        # Search existing run by rel_dir or data_dir, falling back to run_name
        existing_rid = None
        client = mlflow.tracking.MlflowClient()
        filters = [f"tags.rel_dir = '{rel_dir}'"] if rel_dir else []
        if data_dir:
            filters.append(f"tags.data_dir = '{data_dir}'")
        if run_name:
            filters.extend([f"tags.run_name = '{run_name}'", f"attributes.run_name = '{run_name}'"])

        for filt in filters:
            with suppress(Exception):
                runs = client.search_runs(experiment_ids=[experiment_id], filter_string=filt, max_results=1)
                if runs:
                    existing_rid = runs[0].info.run_id
                    break

        if mlflow.active_run() is not None:
            with suppress(Exception):
                mlflow.end_run()

        if existing_rid:
            self._active_run = mlflow.start_run(run_id=existing_rid, tags=run_tags)
        else:
            self._active_run = mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=run_tags)

        target_rel = run_tags.get("rel_dir") or run_name
        if target_rel and "data_dir" not in run_tags:
            p = str(self.resolver.get_experiment_dir(target_rel, create=True).resolve())
            with suppress(Exception):
                mlflow.set_tags({"data_dir": p, "rel_dir": str(target_rel)})
        return self._active_run

    def set_data_dir(self, rel_or_id: str, data_dir: Path | str | None = None) -> None:
        """Explicitly set or update the storage directory tags for the active run."""
        p = str((Path(data_dir) if data_dir else self.resolver.get_experiment_dir(rel_or_id, create=True)).resolve())
        with suppress(Exception):
            mlflow.set_tags({"data_dir": p, "rel_dir": str(rel_or_id)})

    @property
    def client(self) -> Any:
        return mlflow.tracking.MlflowClient()

    @property
    def active_run_id(self) -> str | None:
        if self._active_run is not None:
            return getattr(self._active_run.info, "run_id", None)
        active = mlflow.active_run()
        return active.info.run_id if active is not None else None

    def log_params(self, params: dict[str, Any]) -> None:
        safe = {
            k: (
                "None"
                if v is None
                else str(v)[:500]
                if isinstance(v, (list, tuple, dict)) or len(str(v)) > 500
                else v
            )
            for k, v in params.items()
        }
        active = self._active_run or mlflow.active_run()
        if active is not None:
            with suppress(Exception):
                run = self.client.get_run(active.info.run_id)
                existing = run.data.params
                new_params = {k: v for k, v in safe.items() if k not in existing}
                changed = {
                    k: str(v)
                    for k, v in safe.items()
                    if k in existing and str(existing[k]) != str(v)
                }
                if changed:
                    mlflow.set_tags(changed)
                safe = new_params
        if safe:
            with suppress(Exception):
                mlflow.log_params(safe)

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
                    tags=safe_tags,
                    request_preview=_truncate_preview(request_preview),
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
            request_preview=_truncate_preview(request_preview),
            response_preview=_truncate_preview(response_preview),
        )

    def log_expectation(
        self, trace_id: str, name: str, value: Any, source_id: str = "ground_truth"
    ) -> None:
        """Log expectation (ground truth) assessment on a trace."""
        with suppress(Exception):
            self._flush_before_assessment(trace_id)
            source = AssessmentSource(
                source_type=AssessmentSourceType.CODE, source_id=source_id
            )
            mlflow.log_expectation(
                trace_id=trace_id, name=name, value=value, source=source
            )

    def log_feedback(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "evaluator",
        rationale: str | None = None,
    ) -> None:
        """Log feedback assessment on a trace."""
        with suppress(Exception):
            self._flush_before_assessment(trace_id)
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

    def _flush_before_assessment(self, trace_id: str) -> None:
        if trace_id not in self._assessment_ready_traces:
            self.flush_traces()
            self._assessment_ready_traces.add(trace_id)

    def flush_traces(self) -> None:
        """Flush background async trace queue."""
        mlflow.flush_trace_async_logging()


# Re-export analysis tracking helper
from .analysis_tracking import log_analysis_results_to_mlflow  # noqa: E402

__all__ = ["MLflowTracker", "log_analysis_results_to_mlflow"]
