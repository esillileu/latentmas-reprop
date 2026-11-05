"""Artifacts management and summary reporting for single intervention analysis."""

import json
from pathlib import Path
from typing import Any

from ...domain.services.prompts import initial_input
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
        avg_len = sum(r.get("generated_length", 0) for r in cond_records) / max(
            1, count
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
    comparison_id: str = "",
    result_fingerprint: str = "",
    trace_tags: dict[str, Any] | None = None,
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
            (
                str(r["gold"])
                for r in cond_records.values()
                if r.get("gold") is not None
            ),
            "",
        )
    sample_id = f"sample_{skey}"
    trace_id: str | None = None

    with tracker.start_sample_trace(
        name=sample_id,
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
            "initial_input": str(shared_meta.get("initial_input", initial_input(args))),
            "sample_id": sample_id,
            "sample_key": skey,
            "comparison_id": comparison_id,
            "result_fingerprint": result_fingerprint,
            "latent_steps": str(getattr(args, "latent_steps", 10)),
            "seed": str(getattr(args, "seed", 42)),
            "git_commit": git_hash,
            "recording": "saved_results",
            **(trace_tags or {}),
        },
        request_preview=f"[{sample_id}] {question[:90]}",
    ) as root_span:
        with tracker.start_span(
            name="initial_states"
            if "initial_states" in shared_meta
            else "common_latent",
            span_type="CHAIN",
            inputs={"initial_states": shared_meta["initial_states"]}
            if "initial_states" in shared_meta
            else {
                "sample_key": skey,
                "initial_input": shared_meta.get("initial_input", initial_input(args)),
                "prompt": shared_meta.get("prompt", shared_meta.get("prompt_text", "")),
                "prompt_token_ids": shared_meta.get("prompt_token_ids", []),
                "initial_role": shared_meta.get("initial_role"),
                "target_latent_steps": getattr(args, "latent_steps", 10),
                "latent_steps": shared_meta.get("latent_steps", 0),
            },
        ) as b_span:
            if b_span is not None:
                b_span.set_outputs(
                    {
                        "prefills": {
                            role: {
                                "prompt_length": meta["prompt_length"],
                                "latent_steps": meta["latent_steps"],
                            }
                            for role, meta in shared_meta["initial_states"].items()
                        }
                    }
                    if "initial_states" in shared_meta
                    else {
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
                    "intervention_token_ids": rec.get("intervention_token_ids", []),
                    "step": rec.get("identity", {}).get("step"),
                    "baseline": cond in {"QE", "RQE", "PE", "BASELINE"},
                    "initial_role": rec.get("initial_role"),
                    "receiver_role": rec.get("receiver_role"),
                    "initial_prompt": rec.get("initial_prompt"),
                    "initial_prompt_token_ids": rec.get("initial_prompt_token_ids"),
                    "latent_steps_executed": rec.get("latent_steps_executed"),
                    "stages": rec.get("stages", []),
                    "rng_provenance": rec.get("rng_provenance"),
                    "sampling": rec.get("identity", {}).get("sampling"),
                    "max_new_tokens": rec.get("identity", {}).get("max_new_tokens"),
                },
            ) as c_span:
                if c_span is not None:
                    g_tok = int(rec.get("generated_length", 0))
                    prompt_tokens = interv_len
                    if "initial_states" in shared_meta:
                        prompt_tokens += len(rec["initial_prompt_token_ids"])
                    c_span.set_token_usage(
                        prompt_tokens=prompt_tokens, completion_tokens=g_tok
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
                for stage in rec.get("stages", []):
                    with tracker.start_span(
                        name=stage["role"],
                        span_type="CHAIN",
                        inputs={
                            "prompt": stage["prompt"],
                            "token_ids": stage["token_ids"],
                            "latent_steps": stage["latent_steps"],
                        },
                    ) as stage_span:
                        if stage_span is not None:
                            stage_span.set_outputs(
                                {
                                    key: stage[key]
                                    for key in (
                                        "cache_length_before",
                                        "cache_length_after",
                                        "hidden_offset",
                                        "hidden_length",
                                    )
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
