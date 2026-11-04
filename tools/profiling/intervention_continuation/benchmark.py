"""Benchmark and Equivalence Verification Suite for Decoder Variants.

Compares:
1. Equivalence: Output token IDs and hidden states against baseline.
2. Latency & Throughput: Token-by-token latency, overall tokens/s, decoding latency.
3. Memory: Peak GPU memory.
4. Generates comparison tables and Markdown reports.
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from latentmas_reprop.domain.services.kv_cache import clone_past_kv
from tools.profiling.intervention_continuation.equivalence import compare_equivalence
from tools.profiling.intervention_continuation.report import generate_markdown_report
from tools.profiling.intervention_continuation.variants import VARIANTS


def prepare_benchmark_context(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: torch.device,
) -> tuple[Any, torch.Tensor]:
    """Generates initial KV cache and initial hidden state from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
    initial_hidden = outputs.hidden_states[-1][0, -1, :]
    return outputs.past_key_values, initial_hidden


def run_benchmark_trial(
    decoder_fn: Any,
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_runs: int = 3,
) -> dict[str, Any]:
    """Run decoding with warmup and multiple synchronized iterations."""
    # Warmup run
    kv_warmup = clone_past_kv(past_kv)
    _ = decoder_fn(
        model=model,
        device=device,
        past_kv=kv_warmup,
        initial_hidden=initial_hidden.clone(),
        tokenizer=tokenizer,
        max_new_tokens=min(16, max_new_tokens),
        temperature=temperature,
        top_p=top_p,
        report_progress=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    token_counts = []
    generated_ids_list = []
    hiddens_list = []

    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        kv_run = clone_past_kv(past_kv)
        t_start = time.perf_counter()

        ids, hiddens, _reason, _exhausted = decoder_fn(
            model=model,
            device=device,
            past_kv=kv_run,
            initial_hidden=initial_hidden.clone(),
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            report_progress=False,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_elapsed = time.perf_counter() - t_start

        times.append(t_elapsed)
        token_counts.append(len(ids))
        generated_ids_list.append(ids)
        hiddens_list.append(hiddens)

    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    avg_tokens = float(np.mean(token_counts))
    throughput = avg_tokens / avg_time if avg_time > 0 else 0.0
    token_latency_ms = (avg_time / avg_tokens * 1000) if avg_tokens > 0 else 0.0

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    return {
        "avg_time_s": avg_time,
        "std_time_s": std_time,
        "tokens_generated": int(avg_tokens),
        "throughput_tok_s": throughput,
        "per_token_latency_ms": token_latency_ms,
        "peak_gpu_mem_mb": peak_mem_mb,
        "sample_ids": generated_ids_list[0],
        "sample_hiddens": hiddens_list[0],
    }


def run_full_benchmark(
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_runs: int = 3,
    device_id: int = 0,
):
    """Run full benchmark comparing baseline and all variants."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print(f"Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(
        device
    )
    model.eval()

    prompt = (
        "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        "market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n"
        "Solution:"
    )

    past_kv, initial_hidden = prepare_benchmark_context(
        model, tokenizer, prompt, device
    )

    # 1. Run baseline
    print("\n>>> Running Baseline (decode_tokens_with_hiddens)...")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    base_results = run_benchmark_trial(
        decoder_fn=VARIANTS["baseline"],
        model=model,
        device=device,
        past_kv=past_kv,
        initial_hidden=initial_hidden,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_runs=num_runs,
    )
    base_ids = base_results.pop("sample_ids")
    base_hiddens = base_results.pop("sample_hiddens")

    print(
        f"Baseline Throughput: {base_results['throughput_tok_s']:.2f} tok/s, "
        f"Latency: {base_results['per_token_latency_ms']:.2f} ms/tok, "
        f"Mem: {base_results['peak_gpu_mem_mb']:.1f} MB"
    )

    all_results = {
        "baseline": {
            **base_results,
            "equivalence": {
                "is_equivalent": True,
                "ids_exact_match": True,
                "max_hidden_diff": 0.0,
            },
        }
    }

    # 2. Run all variants
    for var_name, var_fn in VARIANTS.items():
        if var_name == "baseline":
            continue

        print(f"\n>>> Running Variant: {var_name}...")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        try:
            var_res = run_benchmark_trial(
                decoder_fn=var_fn,
                model=model,
                device=device,
                past_kv=past_kv,
                initial_hidden=initial_hidden,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_runs=num_runs,
            )
            var_ids = var_res.pop("sample_ids")
            var_hiddens = var_res.pop("sample_hiddens")

            # Equivalence check
            eq_check = compare_equivalence(base_ids, base_hiddens, var_ids, var_hiddens)
            speedup = var_res["throughput_tok_s"] / base_results["throughput_tok_s"]
            latency_reduction = (
                (base_results["per_token_latency_ms"] - var_res["per_token_latency_ms"])
                / base_results["per_token_latency_ms"]
                * 100
            )

            var_res["speedup"] = speedup
            var_res["latency_reduction_pct"] = latency_reduction
            var_res["equivalence"] = eq_check

            print(
                f"[{var_name}] Throughput: {var_res['throughput_tok_s']:.2f} tok/s ({speedup:.2f}x), "
                f"Latency: {var_res['per_token_latency_ms']:.2f} ms, "
                f"Equiv: {eq_check['is_equivalent']} (diff={eq_check['max_hidden_diff']:.2e})"
            )

            all_results[var_name] = var_res
        except Exception as e:
            print(f"[{var_name}] FAILED with error: {e}")
            all_results[var_name] = {
                "error": str(e),
                "equivalence": {"is_equivalent": False},
            }

    # Save results to json
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Generate Markdown Report
    report_path = os.path.join(output_dir, "REPORT.md")
    generate_markdown_report(
        all_results, report_path, model_name, max_new_tokens, temperature
    )
    print(f"\nBenchmark completed. Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark intervention decoder variants"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tools/profiling/intervention_continuation/results",
    )
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    run_full_benchmark(
        model_name=args.model,
        output_dir=args.output_dir,
        max_new_tokens=args.tokens,
        temperature=args.temp,
        top_p=args.top_p,
        num_runs=args.runs,
        device_id=args.device_id,
    )


if __name__ == "__main__":
    main()
