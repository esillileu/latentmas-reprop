"""PyTorch Profiler script for intervention continuation decoding.

Analyzes CPU/GPU execution time, CUDA synchronizations, memory allocation,
and top PyTorch/CUDA kernels for decoding steps.
"""

import argparse
import json
import os
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def run_profiler(
    variant_name: str,
    model_name: str,
    output_dir: str,
    max_new_tokens: int = 64,
    device_id: int = 0,
):
    """Run PyTorch Profiler on a given decoder variant."""
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    print(f"[{variant_name}] Loading model: {model_name} on {device}...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(
        device
    )
    model.eval()
    t_model_load = time.perf_counter() - t0

    prompt = (
        "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder at the farmers' "
        "market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n"
        "Solution:"
    )

    t0 = time.perf_counter()
    past_kv, initial_hidden = prepare_benchmark_context(
        model, tokenizer, prompt, device
    )
    t_kv_prep = time.perf_counter() - t0

    decoder_fn = VARIANTS[variant_name]

    # Warmup
    print(f"[{variant_name}] Warming up...")
    try:
        _ = decoder_fn(
            model=model,
            device=device,
            past_kv=past_kv,
            initial_hidden=initial_hidden,
            tokenizer=tokenizer,
            max_new_tokens=8,
            temperature=0.0,
            report_progress=False,
        )
    except Exception as e:
        print(f"[{variant_name}] Warmup error: {e}")
        raise

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    print(f"[{variant_name}] Running PyTorch Profiler for {max_new_tokens} tokens...")
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    trace_file = os.path.join(output_dir, f"trace_{variant_name}.json")
    summary_file = os.path.join(output_dir, f"summary_{variant_name}.txt")

    t_decode_start = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        ids, _hiddens, _reason, _exhausted = decoder_fn(
            model=model,
            device=device,
            past_kv=past_kv,
            initial_hidden=initial_hidden,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            report_progress=False,
        )
    torch.cuda.synchronize()
    t_decode_end = time.perf_counter()
    decode_time = t_decode_end - t_decode_start
    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    tokens_generated = len(ids)
    throughput = tokens_generated / decode_time if decode_time > 0 else 0.0

    print(f"Exporting profiler trace to {trace_file}...")
    try:
        prof.export_chrome_trace(trace_file)
    except Exception as e:
        print(f"Failed to export chrome trace: {e}")

    # Top CUDA kernels / Top CPU operations
    cuda_key_avg = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=30,
    )

    with open(summary_file, "w") as f:
        f.write(f"=== Profiling Summary: {variant_name} ===\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Tokens generated: {tokens_generated}\n")
        f.write(f"Decode time: {decode_time:.4f} s\n")
        f.write(f"Throughput: {throughput:.2f} tokens/s\n")
        f.write(f"Peak GPU memory: {peak_mem_mb:.2f} MB\n")
        f.write(f"Model load time: {t_model_load:.4f} s\n")
        f.write(f"KV prep time: {t_kv_prep:.4f} s\n\n")
        f.write("=== Top Operations ===\n")
        f.write(cuda_key_avg)

    metrics = {
        "variant": variant_name,
        "model": model_name,
        "tokens_generated": tokens_generated,
        "decode_time_s": decode_time,
        "throughput_tok_s": throughput,
        "peak_gpu_mem_mb": peak_mem_mb,
        "model_load_s": t_model_load,
        "kv_prep_s": t_kv_prep,
        "trace_file": trace_file,
    }
    with open(os.path.join(output_dir, f"metrics_{variant_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"[{variant_name}] Done. Throughput: {throughput:.2f} tok/s, Peak Mem: {peak_mem_mb:.1f} MB"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Profile Intervention Continuation Decoder"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=[*list(VARIANTS.keys()), "all"],
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tools/profiling/intervention_continuation/results",
    )
    parser.add_argument("--device_id", type=int, default=0)
    args = parser.parse_args()

    variants = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]
    for v in variants:
        try:
            run_profiler(
                variant_name=v,
                model_name=args.model,
                output_dir=args.output_dir,
                max_new_tokens=args.tokens,
                device_id=args.device_id,
            )
        except Exception as e:
            print(f"Error profiling {v}: {e}")


if __name__ == "__main__":
    main()
