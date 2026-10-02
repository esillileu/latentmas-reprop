"""Measure upstream construction and receiver inference within trace spans."""

from contextlib import nullcontext

from ...domain.ports.tracking_port import DummySpan
from ...domain.services.kv_cache import (
    clone_past_kv,
    estimate_past_kv_bytes,
    get_past_kv_dtype,
    get_past_kv_num_layers,
    get_past_kv_sequence_length,
    truncate_past_kv,
)


def build_upstream(method, item, step, width, context_id, tracker, runtime):
    measurements = {
        "upstream_peak_vram_bytes": 0,
        "upstream_peak_vram_reserved_bytes": 0,
        "upstream_incremental_peak_vram_bytes": 0,
        "upstream_incremental_peak_vram_reserved_bytes": 0,
        "upstream_build_latency_sec": 0.0,
        "upstream_source_sequence_length": 0,
        "upstream_source_cache_bytes": 0,
        "upstream_prompt_tokens": 0,
        "upstream_latent_steps_executed": 0,
        "upstream_agent_count": 0,
        "cache_present": False,
        "cache_bytes": 0,
        "cache_num_layers": 0,
        "cache_dtype": None,
        "handoff_positions": 0,
    }
    if not step:
        return None, [[]], measurements
    span_context = (
        tracker.start_span(
            f"build_upstream_{step}",
            "CHAIN",
            {
                "upstream_latent_steps": step,
                "context_id": context_id,
                "handoff_mode": "full" if width is None else "tail",
                **({"handoff_positions": width} if width is not None else {}),
            },
        )
        if tracker
        else nullcontext(DummySpan())
    )
    with span_context as span:
        with runtime.measure(span) as timing:
            context, traces = method.build_latent_contexts([item])
            measurements.update(
                upstream_source_sequence_length=get_past_kv_sequence_length(context),
                upstream_source_cache_bytes=estimate_past_kv_bytes(context),
                upstream_prompt_tokens=sum(
                    len(a.get("input_ids", [])) for a in traces[0]
                ),
                upstream_latent_steps_executed=sum(
                    a.get("latent_steps", 0) for a in traces[0]
                ),
                upstream_agent_count=len(traces[0]),
            )
            if width is not None:
                context = truncate_past_kv(context, width)
            measurements.update(
                cache_present=context is not None,
                cache_bytes=estimate_past_kv_bytes(context),
                cache_num_layers=get_past_kv_num_layers(context),
                cache_dtype=get_past_kv_dtype(context),
                handoff_positions=get_past_kv_sequence_length(context),
            )
            if width is not None and measurements["handoff_positions"] != width:
                raise ValueError("Upstream context is shorter than handoff width")
        measurements["upstream_build_latency_sec"] = timing["latency_sec"]
        measurements.update(
            {f"upstream_{k}": v for k, v in timing.items() if k != "latency_sec"}
        )
        span.set_attribute("handoff_positions", measurements["handoff_positions"])
        span.set_attribute("handoff_mode", "full" if width is None else "tail")
        span.set_outputs(measurements | {"agents": traces[0]})
    return context, traces, measurements


def decode_receiver(
    method,
    item,
    context,
    traces,
    mode,
    limit,
    step,
    context_id,
    measurements,
    tracker,
    runtime,
):
    span_context = (
        tracker.start_span(
            f"decode_{step}_{mode}",
            "LLM",
            {
                "upstream_latent_steps": step,
                "receiver_mode": mode,
                "context_id": context_id,
                "handoff_positions": measurements["handoff_positions"],
                "max_new_tokens": limit,
            },
        )
        if tracker
        else nullcontext(DummySpan())
    )
    with span_context as span:
        with runtime.measure(span) as timing:
            result = method.decode_with_context(
                [item],
                past_kv=clone_past_kv(context),
                initial_traces=traces,
                receiver_mode=mode,
                max_new_tokens=limit,
            )[0]
        agent = result["agents"][-1]
        diagnostics = {
            **{f"receiver_{k}": v for k, v in timing.items()},
            "receiver_prompt_tokens": len(agent.get("input_ids", [])),
            "receiver_generated_tokens": agent["generated_tokens"],
            "receiver_token_limit_reached": agent["generated_tokens"] >= limit,
            "parse_failure": result["prediction"] is None
            or not str(result["prediction"]).strip(),
            "empty_output": not str(result["raw_prediction"] or "").strip(),
            "execution_success": not bool(result.get("error_msg")),
        }
        diagnostics["receiver_generated_tokens_per_sec"] = (
            agent["generated_tokens"] / timing["latency_sec"]
            if timing["latency_sec"]
            else 0.0
        )
        span.set_inputs(
            {
                "prompt": agent.get("input"),
                "context_id": context_id,
                "upstream_latent_steps": step,
                "receiver_mode": mode,
                "handoff_positions": measurements["handoff_positions"],
                "max_new_tokens": limit,
            }
        )
        span.set_token_usage(
            diagnostics["receiver_prompt_tokens"],
            diagnostics["receiver_generated_tokens"],
        )
        span.set_outputs(
            diagnostics
            | {
                "prediction": result["prediction"],
                "raw_output": result["raw_prediction"],
                "correct": result["correct"],
                "error_msg": result.get("error_msg"),
            }
        )
        if result.get("error_msg"):
            span.set_status("ERROR", result["error_msg"])
    return result, diagnostics
