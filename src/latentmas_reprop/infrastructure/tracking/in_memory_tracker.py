import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ...domain.ports.tracking_port import ExperimentTrackerPort, LiveSpanPort


class InMemorySpan(LiveSpanPort):
    """In-memory span implementation recording inputs, outputs, attributes, and status."""

    def __init__(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        inputs: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> None:
        self.name = name
        self.span_type = span_type
        self.inputs = dict(inputs or {})
        self.outputs: dict[str, Any] = {}
        self.attributes: dict[str, Any] = {}
        self.status: str = "OK"
        self.status_description: str | None = None
        self.token_usage: dict[str, int] = {}
        self._trace_id = trace_id or uuid.uuid4().hex

    def set_inputs(self, inputs: dict[str, Any]) -> None:
        self.inputs.update(inputs)

    def set_outputs(self, outputs: dict[str, Any]) -> None:
        self.outputs.update(outputs)

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def set_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    def set_status(self, status: str, description: str | None = None) -> None:
        self.status = status
        self.status_description = description

    @property
    def trace_id(self) -> str | None:
        return self._trace_id


class InMemoryTracker(ExperimentTrackerPort):
    """High-speed in-memory implementation of ExperimentTrackerPort for unit tests."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self._active_run_id: str | None = None
        self._staging_run_id: str | None = None
        self._staging_experiment_name: str | None = None
        self.traces: list[dict[str, Any]] = []
        self.active_trace: dict[str, Any] | None = None
        self.assessments: list[dict[str, Any]] = []

    @property
    def active_run_id(self) -> str | None:
        return self._active_run_id

    def start_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        run_id = f"in_memory_{uuid.uuid4().hex[:8]}"
        self._active_run_id = run_id
        self.runs[run_id] = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "run_name": run_name or run_id,
            "tags": dict(tags or {}),
            "params": {},
            "metrics": {},
            "artifacts": {},
            "status": "RUNNING",
        }
        return type(
            "RunInfo",
            (),
            {
                "info": type(
                    "Info", (), {"run_id": run_id, "experiment_id": experiment_name}
                )()
            },
        )()

    def log_params(self, params: dict[str, Any]) -> None:
        if self._active_run_id and self._active_run_id in self.runs:
            self.runs[self._active_run_id]["params"].update(params)

    def log_metrics(
        self, metrics: dict[str, float | int], step: int | None = None
    ) -> None:
        if self._active_run_id and self._active_run_id in self.runs:
            self.runs[self._active_run_id]["metrics"].update(metrics)

    def log_artifact(
        self, local_path: Path | str, artifact_path: str | None = None
    ) -> None:
        if self._active_run_id and self._active_run_id in self.runs:
            key = (
                f"{artifact_path}/{Path(local_path).name}"
                if artifact_path
                else Path(local_path).name
            )
            self.runs[self._active_run_id]["artifacts"][key] = str(local_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        if self._active_run_id and self._active_run_id in self.runs:
            self.runs[self._active_run_id]["artifacts"][artifact_file] = dictionary

    def end_run(self, status: str = "FINISHED") -> None:
        if self._active_run_id and self._active_run_id in self.runs:
            self.runs[self._active_run_id]["status"] = status
        self._active_run_id = None

    @property
    def is_staging(self) -> bool:
        return self._staging_run_id is not None

    def start_staging_run(
        self,
        experiment_name: str,
        run_name: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> Any:
        staging_tags = dict(tags or {})
        staging_tags["is_staging"] = "true"
        staging_run_name = (
            f"staging/{run_name}"
            if run_name and not run_name.startswith("staging/")
            else (run_name or "staging/unnamed")
        )
        self._staging_experiment_name = experiment_name
        run = self.start_run(
            experiment_name=experiment_name,
            run_name=staging_run_name,
            tags=staging_tags,
        )
        self._staging_run_id = self.active_run_id
        return run

    def promote_staging_run(
        self,
        final_run_name: str,
        tags: dict[str, Any] | None = None,
        metrics: dict[str, float | int] | None = None,
        params: dict[str, Any] | None = None,
        artifacts: list[tuple[Path | str, str | None]] | None = None,
        status: str = "FINISHED",
    ) -> Any:
        staging_run_id = self._staging_run_id
        experiment_name = self._staging_experiment_name
        self.end_run("FINISHED")

        final_tags = dict(tags or {})
        if staging_run_id:
            final_tags["promoted_from"] = str(staging_run_id)

        final_run = self.start_run(
            experiment_name=experiment_name or "latentmas-reprop",
            run_name=final_run_name,
            tags=final_tags,
        )
        final_run_id = self.active_run_id

        # Re-link traces from staging to final
        if staging_run_id and final_run_id:
            for trace in self.traces:
                if trace.get("run_id") == staging_run_id:
                    trace["run_id"] = final_run_id

        if params:
            self.log_params(params)
        if metrics:
            self.log_metrics(metrics)
        if artifacts:
            for path, art_path in artifacts:
                self.log_artifact(path, artifact_path=art_path)

        self.end_run(status=status)
        if staging_run_id and staging_run_id in self.runs:
            self.runs[staging_run_id]["status"] = "PROMOTED"

        self._staging_run_id = None
        self._staging_experiment_name = None
        return final_run

    def discard_staging_run(self, status: str = "FAILED") -> None:
        self.end_run(status=status)
        self._staging_run_id = None
        self._staging_experiment_name = None

    @contextmanager
    def start_sample_trace(
        self,
        name: str,
        inputs: dict[str, Any],
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        trace_id = uuid.uuid4().hex
        span = InMemorySpan(
            name=name, span_type="CHAIN", inputs=inputs, trace_id=trace_id
        )
        trace_info = {
            "trace_id": trace_id,
            "run_id": self._active_run_id,
            "name": name,
            "inputs": inputs,
            "tags": dict(tags or {}),
            "request_preview": request_preview,
            "response_preview": None,
            "spans": [span],
        }
        self.active_trace = trace_info
        self.traces.append(trace_info)
        try:
            yield span
        finally:
            self.active_trace = None

    @contextmanager
    def start_span(
        self,
        name: str,
        span_type: str = "UNKNOWN",
        inputs: dict[str, Any] | None = None,
    ) -> Generator[LiveSpanPort, None, None]:
        trace_id = self.active_trace["trace_id"] if self.active_trace else None
        span = InMemorySpan(
            name=name, span_type=span_type, inputs=inputs, trace_id=trace_id
        )
        if self.active_trace:
            self.active_trace["spans"].append(span)
        yield span

    def update_current_trace(
        self,
        tags: dict[str, Any] | None = None,
        request_preview: str | None = None,
        response_preview: str | None = None,
    ) -> None:
        if self.active_trace:
            if tags:
                self.active_trace["tags"].update(tags)
            if request_preview is not None:
                self.active_trace["request_preview"] = request_preview
            if response_preview is not None:
                self.active_trace["response_preview"] = response_preview

    def log_expectation(
        self, trace_id: str, name: str, value: Any, source_id: str = "ground_truth"
    ) -> None:
        self.assessments.append(
            {
                "trace_id": trace_id,
                "name": name,
                "value": value,
                "source_id": source_id,
                "kind": "expectation",
            }
        )

    def log_feedback(
        self,
        trace_id: str,
        name: str,
        value: Any,
        source_id: str = "evaluator",
        rationale: str | None = None,
    ) -> None:
        self.assessments.append(
            {
                "trace_id": trace_id,
                "name": name,
                "value": value,
                "source_id": source_id,
                "rationale": rationale,
                "kind": "feedback",
            }
        )

    def flush_traces(self) -> None:
        pass
