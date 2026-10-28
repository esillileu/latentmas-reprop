"""Artifacts management and summary reporting for single intervention analysis."""

import contextlib
import json
from pathlib import Path
from typing import Any

from ...domain.models.use_cases import (
    INTERVENTION_CONTINUATION,
    mlflow_experiment_name,
)
from ...domain.services.prompts import resolve_p1_variant_id
from ...infrastructure.storage import save_json_atomic


def build_summary_dict(
    records: list[dict[str, Any]],
    conditions: list[str],
    total_samples: int,
    config_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate metrics and counts across all completed samples and conditions."""
    metrics: dict[str, Any] = {
        "sample_count": total_samples,
        "total": total_samples,
    }
    cond_stats: dict[str, Any] = {}
    for cond in conditions:
        cond_records = [r for r in records if r.get("condition") == cond]
        count = len(cond_records)
        correct = sum(1 for r in cond_records if r.get("correct") is True)
        acc = correct / max(1, count)
        exhausted = sum(1 for r in cond_records if r.get("budget_exhausted") is True)
        avg_len = (
            sum(r.get("generated_length", 0) for r in cond_records) / max(1, count)
        )
        cond_stats[cond] = {
            "completed": count,
            "correct": correct,
            "accuracy": acc,
            "budget_exhausted": exhausted,
            "exhausted_rate": exhausted / max(1, count),
            "avg_generated_length": avg_len,
        }
        metrics[f"{cond}_accuracy"] = acc
        metrics[f"{cond}_correct"] = correct
        metrics[f"{cond}_completed"] = count

    return {
        "status": "complete",
        "sample_count": total_samples,
        "conditions": conditions,
        "condition_statistics": cond_stats,
        "metrics": metrics,
        "config": config_dict or {},
    }


def save_experiment_summary(
    exp_dir: Path,
    summary_data: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    """Save summary.json and samples.jsonl atomically at the experiment root."""
    save_json_atomic(exp_dir / "summary.json", summary_data)
    jsonl_path = exp_dir / "samples.jsonl"
    tmp_jsonl = jsonl_path.with_name(f"{jsonl_path.name}.tmp")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_jsonl.replace(jsonl_path)


def start_intervention_continuation_tracker(
    tracker: Any,
    exp_dir: Path,
    exp_relpath: str,
    git_hash: str,
    args: Any,
) -> bool:
    """Start the MLflow run and log experiment parameters before the sample loop."""
    if not tracker:
        return False
    exp_name = mlflow_experiment_name(INTERVENTION_CONTINUATION)
    p1_var = resolve_p1_variant_id(args)
    with contextlib.suppress(Exception):
        tracker.start_run(
            experiment_name=exp_name,
            run_name=exp_relpath,
            tags={
                "model": str(getattr(args, "model_name", "")),
                "task": str(getattr(args, "task", "gsm8k")),
                "method": str(getattr(args, "method", "latent_mas")),
                "experiment_type": INTERVENTION_CONTINUATION,
                "p1_variant": p1_var,
                "data_dir": str(exp_dir.resolve()),
                "rel_dir": exp_relpath,
                "git_commit": git_hash,
                "run_name": exp_relpath,
            },
        )
        cfg_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        tracker.log_params(
            {
                k: str(v)
                for k, v in cfg_dict.items()
                if isinstance(v, (int, float, bool, str))
            }
        )
        return True
    return False


def trace_intervention_continuation_sample(
    tracker: Any,
    args: Any,
    idx: int,
    skey: str,
    item: dict[str, Any],
    shared_meta: dict[str, Any],
    cond_records: dict[str, dict[str, Any]],
    git_hash: str,
    flush: bool = True,
) -> str | None:
    """Log a per-sample MLflow trace with common_latent and decode_{cond} spans."""
    if not tracker:
        return None
    question = str(item.get("question", shared_meta.get("question", "")))
    if not question and "Question: " in str(shared_meta.get("prompt", "")):
        question = (
            str(shared_meta["prompt"]).split("Question: ", 1)[1].split("\n\n", 1)[0]
        )
    gold = str(item.get("gold", shared_meta.get("gold", "")))
    if not gold:
        gold = next(
            (str(r["gold"]) for r in cond_records.values() if r.get("gold") is not None),
            "",
        )
    sample_id = f"sample_{idx}"
    trace_id: str | None = None

    with contextlib.suppress(Exception):
        with tracker.start_sample_trace(
            name="intervention_continuation_sample",
            inputs={
                "sample_index": idx,
                "sample_key": skey,
                "question": question,
                "gold": gold,
            },
            tags={
                "task": str(getattr(args, "task", "gsm8k")),
                "model": str(getattr(args, "model_name", "")),
                "method": str(getattr(args, "method", "latent_mas")),
                "p1_variant": str(shared_meta.get("p1_variant", resolve_p1_variant_id(args))),
                "sample_id": sample_id,
                "sample_key": skey,
                "latent_steps": str(getattr(args, "latent_steps", 10)),
                "seed": str(getattr(args, "seed", 42)),
                "git_commit": git_hash,
            },
            request_preview=f"[{sample_id}] {question[:90]}",
        ) as root_span:
            with tracker.start_span(
                name="common_latent",
                span_type="CHAIN",
                inputs={
                    "sample_key": skey,
                    "p1_variant": shared_meta.get("p1_variant", resolve_p1_variant_id(args)),
                    "prompt": shared_meta.get("prompt", shared_meta.get("prompt_text", "")),
                    "latent_steps": shared_meta.get("latent_steps", getattr(args, "latent_steps", 10)),
                },
            ) as b_span:
                if b_span is not None:
                    b_span.set_outputs(
                        {
                            "prompt_length": shared_meta.get("prompt_length", 0),
                            "latent_steps": shared_meta.get("latent_steps", 0),
                        }
                    )

            for cond, rec in cond_records.items():
                interv_text = (
                    rec.get("intervention_signal")
                    or rec.get("intervention_prompt")
                    or rec.get("intervention_text")
                )
                interv_len = int(
                    rec.get(
                        "intervention_length",
                        rec.get("intervention_token_count", 0),
                    )
                )
                with tracker.start_span(
                    name=f"decode_{cond}",
                    span_type="LLM",
                    inputs={
                        "condition": cond,
                        "sample_key": skey,
                        "intervention_type": rec.get("intervention_type", ""),
                        "intervention_text": interv_text,
                        "intervention_length": interv_len,
                    },
                ) as c_span:
                    if c_span is not None:
                        g_tok = int(rec.get("generated_length", 0))
                        c_span.set_token_usage(
                            prompt_tokens=interv_len, completion_tokens=g_tok
                        )
                        c_span.set_outputs(
                            {
                                "condition": cond,
                                "prediction": rec.get("prediction"),
                                "gold": rec.get("gold", gold),
                                "correct": bool(rec.get("correct", False)),
                                "generated_text": rec.get("generated_text", ""),
                                "generated_length": g_tok,
                                "finish_reason": rec.get("finish_reason", ""),
                                "budget_exhausted": bool(
                                    rec.get("budget_exhausted", False)
                                ),
                            }
                        )

            if root_span is not None:
                root_out: dict[str, Any] = {
                    "sample_index": idx,
                    "sample_key": skey,
                }
                for cond, rec in cond_records.items():
                    root_out[f"{cond}_prediction"] = rec.get("prediction")
                    root_out[f"{cond}_correct"] = bool(rec.get("correct", False))
                root_span.set_outputs(root_out)
                resp = " | ".join(
                    f"{c}: {str(r.get('prediction'))[:40]} "
                    f"({'PASS' if r.get('correct') else 'FAIL'})"
                    for c, r in cond_records.items()
                )
                tracker.update_current_trace(response_preview=resp)
                trace_id = getattr(root_span, "trace_id", None)

        if trace_id:
            tracker.log_expectation(
                trace_id=trace_id,
                name="expected_answer",
                value=gold,
                source_id="ground_truth",
            )
            for cond, rec in cond_records.items():
                tracker.log_feedback(
                    trace_id=trace_id,
                    name=f"{cond}_correct",
                    value=bool(rec.get("correct", False)),
                    source_id="evaluator",
                )
        if flush:
            tracker.flush_traces()
    return trace_id


def log_to_mlflow_tracker(
    tracker: Any,
    exp_dir: Path,
    exp_relpath: str,
    git_hash: str,
    args: Any,
    summary_data: dict[str, Any],
    run_already_started: bool = False,
) -> None:
    """Log lightweight summary metrics, parameters, tags, and small artifacts to MLflow."""
    if not tracker:
        return
    with contextlib.suppress(Exception):
        if not run_already_started:
            start_intervention_continuation_tracker(
                tracker, exp_dir, exp_relpath, git_hash, args
            )
        metrics = summary_data.get("metrics", {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                tracker.log_metric(k, float(v))

        if (exp_dir / "summary.json").is_file():
            tracker.log_artifact(str(exp_dir / "summary.json"))
        if (exp_dir / "config.json").is_file():
            tracker.log_artifact(str(exp_dir / "config.json"))
        if (exp_dir / "samples.jsonl").is_file():
            tracker.log_artifact(str(exp_dir / "samples.jsonl"))

        tracker.flush_traces()
        tracker.end_run("FINISHED")
