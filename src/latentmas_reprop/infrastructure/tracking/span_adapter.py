import secrets
from contextlib import suppress
from typing import Any

from mlflow.tracing.constant import SpanAttributeKey, TokenUsageKey

from ...domain.ports.tracking_port import LiveSpanPort

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
