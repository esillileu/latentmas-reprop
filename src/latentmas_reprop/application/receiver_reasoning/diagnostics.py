"""Descriptive execution diagnostics shared by run logging and offline analysis."""

from statistics import mean, median

RECORD_DIAGNOSTICS = (
    "receiver_latency_sec",
    "receiver_prompt_tokens",
    "upstream_build_latency_sec",
    "upstream_prompt_tokens",
    "upstream_latent_steps_executed",
    "cache_bytes",
    "cache_num_layers",
    "cache_dtype",
    "cache_present",
    "handoff_positions",
    "upstream_source_sequence_length",
    "upstream_source_cache_bytes",
    "parse_failure",
    "empty_output",
    "execution_success",
    "trace_id",
    "upstream_peak_vram_bytes",
    "upstream_peak_vram_reserved_bytes",
    "receiver_peak_vram_bytes",
    "receiver_peak_vram_reserved_bytes",
    "sample_peak_vram_bytes",
    "sample_peak_vram_reserved_bytes",
    "receiver_incremental_peak_vram_bytes",
    "receiver_incremental_peak_vram_reserved_bytes",
    "upstream_incremental_peak_vram_bytes",
    "upstream_incremental_peak_vram_reserved_bytes",
    "sample_incremental_peak_vram_bytes",
    "sample_incremental_peak_vram_reserved_bytes",
)


def numeric_statistics(values):
    return (
        {
            "total": sum(values),
            "mean": mean(values),
            "median": median(values),
            "min": min(values),
            "max": max(values),
        }
        if values
        else {}
    )


def cell_diagnostics(rows):
    """Leave unrecorded historical measurements absent rather than invent zeros."""
    result = {}
    for field in (
        "receiver_latency_sec",
        "receiver_prompt_tokens",
        "upstream_build_latency_sec",
        "upstream_prompt_tokens",
        "upstream_latent_steps_executed",
        "upstream_agent_count",
        "upstream_source_sequence_length",
        "upstream_source_cache_bytes",
        "handoff_positions",
        "cache_bytes",
        "cache_num_layers",
        "upstream_peak_vram_bytes",
        "upstream_peak_vram_reserved_bytes",
        "receiver_peak_vram_bytes",
        "receiver_peak_vram_reserved_bytes",
        "sample_peak_vram_bytes",
        "sample_peak_vram_reserved_bytes",
        "receiver_incremental_peak_vram_bytes",
        "receiver_incremental_peak_vram_reserved_bytes",
        "upstream_incremental_peak_vram_bytes",
        "upstream_incremental_peak_vram_reserved_bytes",
        "sample_incremental_peak_vram_bytes",
        "sample_incremental_peak_vram_reserved_bytes",
    ):
        values = [r[field] for r in rows if field in r]
        if values:
            result[field] = numeric_statistics(values)
    latencies = result.get("receiver_latency_sec", {})
    if latencies.get("total", 0) > 0:
        result["generated_tokens_per_sec"] = (
            sum(
                r["receiver_generated_tokens"]
                for r in rows
                if "receiver_latency_sec" in r
            )
            / latencies["total"]
        )
    return result


def inference_costs(records):
    contexts = {r["context_id"]: r for r in records}
    upstream = [r for r in contexts.values() if r["upstream_latent_steps"] > 0]
    costs = {
        "runtime_context_build_sec": sum(
            r["upstream_build_latency_sec"] for r in upstream
        ),
        "runtime_receiver_sec": sum(r["receiver_latency_sec"] for r in records),
        "output_tokens_total": sum(r["receiver_generated_tokens"] for r in records),
        "receiver_prompt_tokens_total": sum(
            r["receiver_prompt_tokens"] for r in records
        ),
        "upstream_prompt_tokens_total": sum(
            r["upstream_prompt_tokens"] for r in upstream
        ),
        "latent_steps_total": sum(
            r["upstream_latent_steps_executed"] for r in upstream
        ),
    }
    costs["inference_time_sec"] = (
        costs["runtime_context_build_sec"] + costs["runtime_receiver_sec"]
    )
    costs["generated_tokens_per_sec"] = (
        costs["output_tokens_total"] / costs["runtime_receiver_sec"]
        if costs["runtime_receiver_sec"]
        else 0.0
    )
    return costs


def execution_counts(records, expected):
    return {
        "n_records_expected": expected,
        "n_records_completed": len(records),
        "n_records_successful": sum(r["execution_success"] for r in records),
        "n_errors": sum(not r["execution_success"] for r in records),
        "parse_failure_count": sum(r["parse_failure"] for r in records),
        "empty_output_count": sum(r["empty_output"] for r in records),
        "token_limit_count": sum(r["receiver_token_limit_reached"] for r in records),
    }
