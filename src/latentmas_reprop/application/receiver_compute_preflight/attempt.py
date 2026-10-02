"""Trace actual generation attempts without including export time in receiver cost."""

from contextlib import nullcontext

from ...domain.ports.tracking_port import DummySpan


def generate_attempt(
    generate,
    method,
    item,
    context,
    cap,
    budgets,
    purpose,
    attempt_index,
    tracker,
    runtime,
    trace_inputs,
    termination,
):
    name = f"{(trace_inputs or {}).get('handoff_condition', 'receiver')}_{purpose}_attempt_{attempt_index}_cap_{cap}"
    inputs = (trace_inputs or {}) | {
        "question": item["question"],
        "gold": item.get("gold"),
        "model": (trace_inputs or {}).get(
            "model", getattr(method.model, "model_name", None)
        ),
        "max_new_tokens": cap,
        "purpose": purpose,
        "temperature": 0.0,
        "top_p": 1.0,
        "receiver_mode": "free",
    }
    span_context = (
        tracker.start_span(name, "LLM", inputs) if tracker else nullcontext(DummySpan())
    )
    with span_context as span:
        measured = {}
        measurement = runtime.measure(span) if runtime else nullcontext({})
        try:
            with measurement as measured:
                ids, metadata = generate(
                    method, item, context, cap, budgets, purpose == "free"
                )
        except (Exception, KeyboardInterrupt) as exc:
            span.set_status("ERROR", f"{type(exc).__name__}: {exc}")
            span.set_outputs(
                {"execution_success": False, "error_msg": str(exc), "runtime": measured}
            )
            raise
        span.set_inputs(
            {
                "prompt": metadata["receiver_prompt"],
                "parameters": {"max_new_tokens": cap, "temperature": 0.0, "top_p": 1.0},
            }
        )
        for key, value in inputs.items():
            if value is not None:
                span.set_attribute(key, value)
        span.set_attribute("input_token_ids", metadata["receiver_input_ids"])
        span.set_token_usage(metadata["receiver_prompt_tokens"], len(ids))
        metadata.update(
            {f"receiver_{k}": v for k, v in measured.items() if k != "latency_sec"}
        )
        metadata["receiver_execution_latency_sec"] = measured.get("latency_sec")
        metadata["receiver_trace_id"] = span.trace_id
        metadata["receiver_span_name"] = name
        reason = termination(method.model, ids, cap)
        span.set_outputs(
            {
                "text": method.model.tokenizer.decode(ids, skip_special_tokens=True),
                "termination_reason": reason,
            }
        )
        span.set_attribute("generated_token_ids", ids)
        span.set_attribute("termination_reason", reason)
        span.set_attribute("execution_success", True)
        for key, value in metadata.items():
            if (
                key not in {"receiver_prompt", "receiver_input_ids"}
                and value is not None
            ):
                span.set_attribute(key, value)
        return ids, metadata


def finish_attempt(metadata, ids, cap, reason, method):
    return {
        "cap": cap,
        "generated_token_ids": ids,
        "generated_tokens": len(ids),
        "termination_reason": reason,
        "receiver_latency_sec": metadata["receiver_latency_sec"],
        "receiver_trace_id": metadata["receiver_trace_id"],
        "receiver_span_name": metadata["receiver_span_name"],
        "receiver_prompt": metadata["receiver_prompt"],
        "receiver_input_ids": metadata["receiver_input_ids"],
        "receiver_thinking_open": metadata["receiver_thinking_open"],
        "raw_receiver_output": method.model.tokenizer.decode(
            ids, skip_special_tokens=True
        ),
    }
