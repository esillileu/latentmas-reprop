from contextlib import suppress
from typing import Any

import mlflow
from mlflow.entities.assessment_source import AssessmentSource, AssessmentSourceType


def record_trace_expectation(
    trace_id: str,
    name: str,
    value: Any,
    source_id: str = "ground_truth",
) -> None:
    """Record an expectation assessment on an MLflow trace."""
    with suppress(Exception):
        source = AssessmentSource(
            source_type=AssessmentSourceType.CODE, source_id=source_id
        )
        mlflow.log_expectation(trace_id=trace_id, name=name, value=value, source=source)


def record_trace_feedback(
    trace_id: str,
    name: str,
    value: Any,
    source_id: str = "evaluator",
    rationale: str | None = None,
) -> None:
    """Record a feedback assessment on an MLflow trace."""
    with suppress(Exception):
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
