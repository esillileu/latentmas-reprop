"""Artifact persistence and MLflow artifact logging for intervention experiments."""

import contextlib
from pathlib import Path
from typing import Any

from ...domain.models import InterventionMetrics, SampleInterventionRecord
from ...domain.ports.cache_port import CacheLayer, CachePort
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ..common.reporter import clean_config_args, write_json, write_jsonl, write_yaml


def start_intervention_tracker(
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    model: Any,
    n_samples: int,
    conditions: list[str],
    git_hash: str,
) -> None:
    """Start tracking run and log run parameters if tracker_port is available."""
    if not tracker_port:
        return
    exp_name = getattr(args, "tracking_experiment_name", "latentmas_intervention")
    run_name = f"{args.model_name}_{args.task}_intervention"
    backend_name = "vllm" if getattr(args, "use_vllm", False) else "transformers"
    model_dtype = (
        str(getattr(model.model, "dtype", "unknown"))
        if hasattr(model, "model")
        else "unknown"
    )
    tracker_port.start_run(
        experiment_name=exp_name,
        run_name=run_name,
        tags={
            "model": args.model_name,
            "task": args.task,
            "method": args.method,
            "experiment_type": "intervention",
            "git_commit": git_hash,
        },
    )
    tracker_port.log_params(
        {
            "model": args.model_name,
            "task": args.task,
            "split": getattr(args, "split", "test"),
            "sample_count": n_samples,
            "method": args.method,
            "prompt": getattr(args, "prompt", "sequential"),
            "latent_steps": args.latent_steps,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "backend": backend_name,
            "dtype": model_dtype,
            "cross_pairing_policy": getattr(args, "cross_policy", "shift_1"),
            "git_commit": git_hash,
            "intervention_conditions": str(conditions),
            "device": str(model.device),
        }
    )


def log_sample_assessments(
    tracker_port: ExperimentTrackerPort | None,
    trace_id: str | None,
    item: dict,
    gold: str,
    conditions: list[str],
    cond_preds: dict[str, tuple[str | None, bool, str]],
    sample_records_for_sample: dict[str, SampleInterventionRecord],
) -> None:
    """Log GenAI expectations and feedback assessments for a sample trace."""
    if not tracker_port or not trace_id:
        return

    tracker_port.log_expectation(
        trace_id=trace_id,
        name="expected_answer",
        value=gold,
        source_id="ground_truth",
    )
    solution_text = item.get("solution")
    if solution_text:
        tracker_port.log_expectation(
            trace_id=trace_id,
            name="reference_solution",
            value=str(solution_text),
            source_id="ground_truth",
        )
    sample_has_error = any(
        rec.error is not None for rec in sample_records_for_sample.values()
    )
    tracker_port.log_feedback(
        trace_id=trace_id,
        name="execution_success",
        value=not sample_has_error,
        source_id="default",
        rationale="sample status=success"
        if not sample_has_error
        else "sample status=error",
    )
    for cond in conditions:
        if cond in cond_preds:
            _, c_ok, _ = cond_preds[cond]
            tracker_port.log_feedback(
                trace_id=trace_id,
                name=f"{cond}_correct",
                value=c_ok,
                source_id="evaluator",
            )
    if "own" in cond_preds:
        own_p, own_c, _ = cond_preds["own"]
        tracker_port.log_feedback(
            trace_id=trace_id,
            name="runtime_correctness",
            value=own_c,
            source_id="default",
            rationale=f"normalized prediction='{own_p}'; expected='{gold}'",
        )


def save_intervention_artifacts(
    cache_port: CachePort,
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    run_stem: str,
    metrics: InterventionMetrics,
    sample_records: list[SampleInterventionRecord],
    paired_transitions: dict[str, Any],
    harm_rescue: dict[str, dict[str, int]],
    correctness_patterns: dict[str, int],
    mcnemar_dict: dict[str, Any],
    cache_seq_stats: dict[str, Any],
    ram_stats: dict[str, Any],
) -> None:
    """Save all evaluation artifacts to disk and log to MLflow tracker."""
    local_cache_file = f"{run_stem}.json"
    with contextlib.suppress(Exception):
        cache_port.save_json(
            CacheLayer.EVALUATION_INTERVENTIONS,
            local_cache_file,
            {
                "metrics": metrics.to_dict(),
                "records": [r.to_dict() for r in sample_records],
            },
        )

    runtime_dir = Path(cache_port.get_layer_path(CacheLayer.RUNTIME))
    sample_results_path = runtime_dir / f"{run_stem}_sample_results.jsonl"
    summary_path = runtime_dir / f"{run_stem}_summary.json"
    resolved_config_path = runtime_dir / f"{run_stem}_resolved_config.yaml"
    patterns_path = runtime_dir / f"{run_stem}_correctness_patterns.json"

    write_jsonl(sample_results_path, [r.to_dict() for r in sample_records])

    clean_args = clean_config_args(args)
    write_json(
        summary_path,
        {
            "metrics": metrics.to_dict(),
            "paired_transitions": paired_transitions,
            "harm_rescue": harm_rescue,
            "correctness_patterns": correctness_patterns,
            "mcnemar_tests": mcnemar_dict,
            "cache_seq_len_stats": cache_seq_stats,
            "ram_usage_stats": ram_stats,
            "config": clean_args,
        },
    )

    write_yaml(resolved_config_path, clean_args)
    write_json(patterns_path, correctness_patterns)

    if tracker_port:
        tracker_port.log_metrics(metrics.to_mlflow_metrics())
        tracker_port.log_artifact(sample_results_path, artifact_path="results")
        tracker_port.log_artifact(summary_path, artifact_path="results")
        tracker_port.log_artifact(patterns_path, artifact_path="results")
        tracker_port.log_artifact(resolved_config_path, artifact_path="config")
        tracker_port.flush_traces()
        tracker_port.end_run(status="FINISHED")
