"""Render profiling benchmark reports."""

import time
from typing import Any


def generate_markdown_report(
    results: dict[str, Any], path: str, model_name: str, tokens: int, temp: float
):
    """Write markdown summary table and conclusions."""
    with open(path, "w") as f:
        f.write("# Intervention Continuation Decoding Optimization Report\n\n")
        f.write(f"- **Model**: `{model_name}`\n")
        f.write(f"- **Max New Tokens**: {tokens}\n")
        f.write(f"- **Temperature**: {temp} (Greedy)\n")
        f.write(f"- **Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. Summary Comparison Table\n\n")
        f.write(
            "| Variant | Throughput (tok/s) | Speedup | Latency (ms/tok) | Peak GPU Mem (MB) | Exact IDs | Max Hidden Diff | Equivalence |\n"
        )
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        base_thr = results.get("baseline", {}).get("throughput_tok_s", 1.0)
        for name, data in results.items():
            if "error" in data:
                f.write(f"| `{name}` | *Error* | - | - | - | - | - | ❌ Failed |\n")
                continue
            thr = data.get("throughput_tok_s", 0.0)
            speedup = thr / base_thr if base_thr > 0 else 1.0
            lat = data.get("per_token_latency_ms", 0.0)
            mem = data.get("peak_gpu_mem_mb", 0.0)
            eq = data.get("equivalence", {})
            exact_ids = "✅" if eq.get("ids_exact_match", False) else "❌"
            diff = eq.get("max_hidden_diff", 0.0)
            diff_str = f"{diff:.2e}" if diff > 0 else "0.0"
            is_eq = "✅ Equivalent" if eq.get("is_equivalent", False) else "❌ Mismatch"

            f.write(
                f"| `{name}` | {thr:.2f} | **{speedup:.2f}x** | {lat:.2f} | {mem:.1f} | {exact_ids} | {diff_str} | {is_eq} |\n"
            )

        f.write("\n## 2. Key Observations and Bottleneck Analysis\n\n")
        f.write("- **Baseline Bottlenecks**:\n")
        f.write(
            "  1. Per-step GPU-to-CPU synchronization (`curr_hidden.detach().cpu()` and `int(next_id.item())`).\n"
        )
        f.write(
            "  2. Redundant LM Head evaluation inside `model()` when predicting next hidden state.\n"
        )
        f.write(
            "  3. Allocation overhead of attention masks and full hidden state tuples at every step.\n"
        )
        f.write("- **Equivalence Verification**:\n")
        f.write(
            "  - Variants preserving numerical identity produce exact token match and identical hidden representations.\n"
        )
