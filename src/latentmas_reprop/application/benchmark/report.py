"""Summary metrics and environment record for one benchmark run."""

from typing import Any

import torch
import transformers

from ...domain.models import BenchmarkMetrics
from ...infrastructure.models.dtype import peak_allocated_bytes
from ...infrastructure.paths.resolver import get_path_resolver
from .evaluation import (
    OUTPUT_TOKEN_DEFINITION,
    WALL_TIME_DEFINITION,
    summarize_predictions,
)


def build_metrics(
    args: Any,
    predictions: list[dict],
    *,
    run_id: str,
    dtype_name: str,
    eval_seconds: float,
    model_load_seconds: float,
) -> BenchmarkMetrics:
    """Aggregate one run. Eval time excludes model loading."""
    summary = summarize_predictions(predictions)
    total = int(summary["total"])
    per_sample = (eval_seconds / total) if total else 0.0
    return BenchmarkMetrics(
        method=args.method,
        model=args.model_name,
        split=args.split,
        seed=args.seed,
        max_samples=total,
        accuracy=float(summary["accuracy"]),
        correct=int(summary["correct"]),
        total_time_sec=round(eval_seconds, 4),
        time_per_sample_sec=round(per_sample, 4),
        total=total,
        output_tokens_total=int(summary["output_tokens_total"]),
        output_tokens_mean=round(float(summary["output_tokens_mean"]), 4),
        latent_steps_total=int(summary["latent_steps_total"]),
        latent_steps_mean=round(float(summary["latent_steps_mean"]), 4),
        model_load_time_sec=round(model_load_seconds, 4),
        eval_time_sec=round(eval_seconds, 4),
        peak_vram_bytes=peak_allocated_bytes(),
        dtype=dtype_name,
        prompt=args.prompt,
        latent_steps=int(getattr(args, "latent_steps", 0)),
        run_id=run_id,
    )


def runtime_metadata(
    model: Any, dtype_name: str, eval_seconds: float
) -> dict[str, Any]:
    """Record the library, GPU, dtype, and metric definitions used for a run."""
    gpu_name = None
    capability = None
    if torch.cuda.is_available():
        device = getattr(model, "device", torch.device("cuda"))
        index = (
            device.index
            if isinstance(getattr(device, "index", None), int)
            else 0
        )
        gpu_name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        capability = f"{major}.{minor}"
    hf_model = getattr(model, "model", None)
    config = getattr(hf_model, "config", None)
    revision = getattr(config, "_commit_hash", None)
    model_revision = revision if isinstance(revision, str) else None

    model_load_time = getattr(model, "load_time_sec", 0.0)
    try:
        model_load_time_sec = round(float(model_load_time), 4)
    except (TypeError, ValueError):
        model_load_time_sec = 0.0

    return {
        "git_commit": get_path_resolver().get_git_commit_hash(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
        "gpu_capability": capability,
        "dtype": dtype_name,
        "model_revision": model_revision,
        "eval_time_sec": round(eval_seconds, 4),
        "model_load_time_sec": model_load_time_sec,
        "peak_vram_bytes": peak_allocated_bytes(),
        "output_token_definition": OUTPUT_TOKEN_DEFINITION,
        "wall_time_definition": WALL_TIME_DEFINITION,
        "zero_latent_steps_note": (
            "latent_steps=0 runs the non-judger prompts and then drops their KV "
            "cache before the judger. It does not match a run that simply skips "
            "latent iterations while keeping the cache."
        ),
    }
