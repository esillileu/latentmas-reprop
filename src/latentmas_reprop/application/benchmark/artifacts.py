"""Artifact persistence and MLflow tracking for benchmark experiments."""

from pathlib import Path
from typing import Any

from ...domain.models import BenchmarkMetrics
from ...domain.ports.cache_port import CacheLayer, CachePort
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...infrastructure.models.dtype import peak_allocated_bytes
from ..common.reporter import clean_config_args, write_jsonl


def start_benchmark_tracker(
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
) -> None:
    """Start MLflow tracking run and log benchmark parameters."""
    if not tracker_port:
        return

    exp_name = getattr(args, "tracking_experiment_name", None)
    if not exp_name:
        raise ValueError(
            "MLflow tracking experiment name must be specified (--tracking_experiment_name or via config) before running benchmark."
        )

    run_name = f"{args.model_name}_{args.task}_{args.method}"
    backend_name = "vllm" if getattr(args, "use_vllm", False) else "transformers"
    tags = {
        "model": args.model_name,
        "task": args.task,
        "method": args.method,
        "split": getattr(args, "split", "test"),
        "seed": str(args.seed),
        "experiment_type": "benchmark",
    }
    if hasattr(args, "latent_steps") and args.latent_steps:
        tags["latent_steps"] = str(args.latent_steps)
    if hasattr(args, "config_id") and args.config_id:
        tags["config_id"] = str(args.config_id)
    if hasattr(args, "run_id") and args.run_id:
        tags["storage_run_id"] = str(args.run_id)

    tracker_port.start_run(
        experiment_name=exp_name,
        run_name=run_name,
        tags=tags,
    )
    tracker_port.log_params(
        {
            "model": args.model_name,
            "task": args.task,
            "method": args.method,
            "split": getattr(args, "split", "test"),
            "prompt": getattr(args, "prompt", "sequential"),
            "max_samples": args.max_samples,
            "max_new_tokens": getattr(args, "max_new_tokens", 4096),
            "temperature": getattr(args, "temperature", 0.6),
            "top_p": getattr(args, "top_p", 0.95),
            "generate_bs": getattr(args, "generate_bs", 1),
            "latent_steps": getattr(args, "latent_steps", 0),
            "backend": backend_name,
            "seed": args.seed,
        }
    )


def trace_benchmark_sample(
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    problem_idx: int,
    res: dict[str, Any],
) -> None:
    """Record root trace and agent spans for an evaluated problem sample."""
    if not tracker_port:
        return

    q = res.get("question", "").strip()
    gold = str(res.get("gold", ""))
    pred = str(res.get("prediction", ""))
    raw_pred = str(res.get("raw_prediction", ""))
    correct = bool(res.get("correct", False))
    vram_bytes = peak_allocated_bytes()
    vram_gb = round(vram_bytes / (1024**3), 3)
    latency_sec = res.get("latency_sec")

    tags = {
        "problem_idx": str(problem_idx),
        "model": str(args.model_name),
        "method": str(args.method),
        "correct": str(correct),
        "peak_vram_gb": str(vram_gb),
    }
    if latency_sec is not None:
        tags["latency_sec"] = str(latency_sec)

    with tracker_port.start_sample_trace(
        name=f"sample_{problem_idx}",
        inputs={"question": q, "problem_idx": problem_idx},
        tags=tags,
        request_preview=q[:100],
    ) as root_span:
        for a in res.get("agents", []):
            a_name = a.get("name", "Agent")
            a_role = a.get("role", "")
            span_name = f"{a_name}_{a_role}" if a_role else a_name
            with tracker_port.start_span(
                name=span_name,
                span_type="AGENT",
                inputs={
                    "name": a_name,
                    "role": a_role,
                    "prompt": a.get("input", ""),
                    "latent_steps": a.get("latent_steps"),
                },
            ) as agent_span:
                agent_span.set_outputs({"output": a.get("output", "")})

        outputs = {
            "prediction": pred,
            "raw_prediction": raw_pred,
            "correct": correct,
            "peak_vram_gb": vram_gb,
            "peak_vram_bytes": vram_bytes,
        }
        if latency_sec is not None:
            outputs["latency_sec"] = latency_sec
        root_span.set_outputs(outputs)

        trace_id = getattr(root_span, "trace_id", None)
        if trace_id:
            tracker_port.log_expectation(
                trace_id=trace_id,
                name="expected_answer",
                value=gold,
                source_id="ground_truth",
            )
            tracker_port.log_feedback(
                trace_id=trace_id,
                name="correctness",
                value=correct,
                source_id="evaluator",
                rationale=f"prediction='{pred}'; expected='{gold}'",
            )
            tracker_port.log_feedback(
                trace_id=trace_id,
                name="peak_vram_gb",
                value=vram_gb,
                source_id="profiler",
                rationale=f"peak_vram={vram_gb} GB ({vram_bytes} bytes)",
            )
            if latency_sec is not None:
                tracker_port.log_feedback(
                    trace_id=trace_id,
                    name="latency_sec",
                    value=float(latency_sec),
                    source_id="profiler",
                    rationale=f"latency={latency_sec}s",
                )


def save_benchmark_artifacts(
    cache_port: CachePort,
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    run_cache_filename: str,
    metrics: BenchmarkMetrics,
    preds: list[dict[str, Any]],
) -> None:
    """Save benchmark cache and upload artifacts to MLflow."""
    import logging

    logger = logging.getLogger(__name__)

    try:
        cache_port.save_json(
            CacheLayer.EVALUATION_RUNS,
            run_cache_filename,
            {
                "metrics": metrics.to_dict(),
                "predictions": preds,
            },
        )
    except Exception as exc:
        logger.error(
            "Failed to save benchmark run cache %s: %s", run_cache_filename, exc
        )

    if not tracker_port:
        return

    tracker_port.log_metrics(metrics.to_mlflow_metrics())
    clean_args = clean_config_args(args)
    tracker_port.log_dict(
        {
            "metrics": metrics.to_dict(),
            "config": clean_args,
        },
        "results/summary.json",
    )
    tracker_port.log_dict(clean_args, "config/resolved_config.yaml")

    runtime_dir = Path(cache_port.get_layer_path(CacheLayer.RUNTIME))
    stem = run_cache_filename.removesuffix(".json")
    sample_results_path = runtime_dir / f"{stem}_sample_results.jsonl"
    write_jsonl(sample_results_path, preds)
    tracker_port.log_artifact(sample_results_path, artifact_path="results")

    tracker_port.flush_traces()
    tracker_port.end_run(status="FINISHED")
