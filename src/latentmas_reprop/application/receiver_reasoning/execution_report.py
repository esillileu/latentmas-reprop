"""Render experimental execution measurements from exported analysis data."""

from .tables import table


def execution_table(cells):
    fields = {
        "receiver_latency_sec": "mean decode seconds",
        "receiver_prompt_tokens": "mean prompt tokens",
        "upstream_build_latency_sec": "mean shared build seconds",
        "upstream_source_sequence_length": "mean source positions",
        "handoff_positions": "mean handoff positions",
        "upstream_source_cache_bytes": "mean source KV bytes",
        "cache_bytes": "mean handoff KV bytes",
        "cache_num_layers": "mean KV layers",
        "receiver_peak_vram_bytes": "mean receiver peak allocated bytes",
        "receiver_peak_vram_reserved_bytes": "mean receiver peak reserved bytes",
        "upstream_peak_vram_bytes": "mean upstream peak allocated bytes",
        "sample_peak_vram_bytes": "mean sample peak allocated bytes",
    }
    rows = []
    for key, cell in cells.items():
        diagnostics = cell.get("execution_diagnostics", {})
        rows.append(
            [
                key,
                cell.get("error_count"),
                *[diagnostics.get(field, {}).get("mean") for field in fields],
                diagnostics.get("generated_tokens_per_sec"),
            ]
        )
    return table(["cell", "errors", *fields.values(), "generated tokens/sec"], rows)
