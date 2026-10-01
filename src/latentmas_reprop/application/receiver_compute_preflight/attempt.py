"""Trace actual generation attempts without including export time in receiver cost."""

from contextlib import nullcontext

from ...domain.ports.tracking_port import DummySpan
from .cost import compute_proxy
from .inference import evaluate_prefix


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
    evaluator,
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
        span.set_inputs(inputs | metadata)
        span.set_token_usage(metadata["receiver_prompt_tokens"], len(ids))
        metadata.update(
            {f"receiver_{k}": v for k, v in measured.items() if k != "latency_sec"}
        )
        metadata["receiver_execution_latency_sec"] = measured.get("latency_sec")
        metadata["receiver_trace_id"] = span.trace_id
        metadata["receiver_span_name"] = name
        metadata["receiver_generated_tokens_per_sec"] = (
            len(ids) / metadata["receiver_latency_sec"]
            if metadata["receiver_latency_sec"]
            else 0.0
        )
        reason = termination(method.model, ids, cap)
        evaluation = evaluate_prefix(
            method.model.tokenizer,
            evaluator,
            item,
            ids,
            "free",
            reason != "cap",
            metadata["receiver_thinking_open"],
        )
        span.set_attribute("termination_reason", reason)
        span.set_attribute(
            "evaluation_status",
            "pathological"
            if reason == "cap"
            else "no_answer"
            if evaluation["no_answer"]
            else "correct"
            if evaluation["correct"]
            else "incorrect",
        )
        span.set_outputs(
            {
                **evaluation,
                "termination_reason": reason,
                "generated_token_ids": ids,
                "raw_receiver_output": method.model.tokenizer.decode(
                    ids, skip_special_tokens=True
                ),
                "generated_tokens": len(ids),
                "execution_success": True,
                **metadata,
                **compute_proxy(
                    metadata["receiver_cache_positions"],
                    metadata["receiver_prompt_tokens"],
                    len(ids),
                ),
            }
        )
        return ids, metadata


def finish_attempt(metadata, ids, cap, reason, method, item, evaluator):
    evaluation = evaluate_prefix(
        method.model.tokenizer,
        evaluator,
        item,
        ids,
        "free",
        reason != "cap",
        metadata["receiver_thinking_open"],
    )
    return {
        "cap": cap,
        "generated_tokens": len(ids),
        "termination_reason": reason,
        "receiver_latency_sec": metadata["receiver_latency_sec"],
        "receiver_trace_id": metadata["receiver_trace_id"],
        "receiver_span_name": metadata["receiver_span_name"],
        "prediction": evaluation["prediction"],
        "correct": evaluation["correct"],
        "no_answer": evaluation["no_answer"],
        "raw_receiver_output": evaluation["raw_receiver_output"],
    }
