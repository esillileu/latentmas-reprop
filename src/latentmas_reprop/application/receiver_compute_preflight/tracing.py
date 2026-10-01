"""Sample trace lifecycle, rich result spans, and assessment manifests."""

import traceback
from contextlib import contextmanager

from ..common.reporter import build_run_stem

CELL_FIELDS = (
    "prediction",
    "correct",
    "no_answer",
    "final_answer_complete",
    "valid",
    "pathological",
    "generated_tokens",
    "receiver_latency_sec",
    "receiver_attention_pairs",
    "receiver_processed_positions",
    "donor_id",
    "donor_sequence_length",
    "donor_length_delta",
    "donor_length_abs_delta",
    "receiver_trace_id",
    "upstream_trace_id",
    "prefix_verified",
)


def cell_key(row):
    return f"{row['handoff_condition']}_r{row['receiver_budget']}"


def assessments(tracker, state, item, measurements, failure):
    trace_id, names = state["trace_id"], []
    if not trace_id:
        return names
    tracker.log_expectation(trace_id, "expected_answer", item.get("gold", ""))
    names.append("expected_answer")
    if item.get("solution"):
        tracker.log_expectation(trace_id, "reference_solution", str(item["solution"]))
        names.append("reference_solution")
    for name, value, source in (
        ("execution_success", failure is None, "evaluator"),
        ("peak_vram_gib", measurements.get("peak_vram_bytes", 0) / 2**30, "profiler"),
    ):
        tracker.log_feedback(
            trace_id,
            name,
            value,
            source_id=source,
            rationale=failure if name == "execution_success" else None,
        )
        names.append(name)
    for row in state["records"]:
        for field in (
            "correct",
            "no_answer",
            "valid",
            "pathological",
            "generated_tokens",
            "receiver_latency_sec",
            "receiver_attention_pairs",
            "prefix_verified",
        ):
            name = f"{cell_key(row)}_{field}"
            tracker.log_feedback(
                trace_id,
                name,
                row[field],
                rationale=f"prediction={row['prediction']}; gold={item.get('gold')}; error={row.get('error_msg')}",
            )
            names.append(name)
    return names


def run_identity(args, count):
    kind = "smoke" if args.verify_prefix or count <= 2 else "experiment"
    label = getattr(args, "run_label", None) or kind
    stem = build_run_stem(
        f"receiver_compute_preflight_{args.split}_{label}",
        args.task,
        args.model_name,
        count,
    )
    name = f"{stem}_u{'-'.join(map(str, args.upstream_steps))}_r{'-'.join(map(str, args.receiver_budgets))}_seed{args.seed}"
    return name, kind, label


@contextmanager
def sample_trace(
    tracker, runtime, args, config, item, index, sample_id, step, phase, manifest
):
    tags = {
        "experiment_type": "receiver_compute_preflight",
        "model": args.model_name,
        "task": args.task,
        "split": args.split,
        "method": args.method,
        "seed": args.seed,
        "sample_id": sample_id,
        "sample_index": index,
        "upstream_steps": step,
        "phase": phase,
        "handoff_mode": "full",
        "receiver_mode": "free",
        "git_commit": config["git_commit"],
        "version_tag": config.get("version_tag"),
        "run_name": config["run_name"],
        "run_kind": config["run_kind"],
        "run_label": config["run_label"],
        "run_id": config["run_id"],
    }
    state, measurements, failure = None, {}, None
    root_name = f"{args.model_name.split('/')[-1]}_{args.task}_{args.split}_{phase}_sample_{index}_u{step}"
    try:
        with tracker.start_sample_trace(
            root_name,
            {
                "sample_id": sample_id,
                "sample_index": index,
                "question": item["question"],
                "gold": item.get("gold", ""),
                "reference_solution": item.get("solution"),
                "upstream_steps": step,
                "phase": phase,
                "config": config,
            },
            tags=tags,
            request_preview=f"[{index}] U={step} {phase}: {item['question'][:160]}",
        ) as root:
            state = {
                "trace_id": root.trace_id,
                "records": [],
                "outputs": {},
                "required_span_names": [root_name],
            }
            try:
                with runtime.measure(root) as measurements:
                    yield state
            except (Exception, KeyboardInterrupt) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                state["outputs"].update(
                    error_msg=failure, failure_traceback=traceback.format_exc()
                )
                root.set_status("ERROR", failure)
                raise
            finally:
                trajectories = [state["outputs"].get("trajectory", {})] + state[
                    "records"
                ]
                for trajectory in trajectories:
                    for field in ("free_attempts", "prefix_verification_attempts"):
                        for attempt in trajectory.get(field, []):
                            if attempt["receiver_trace_id"] == state["trace_id"]:
                                name = attempt["receiver_span_name"]
                                if name not in state["required_span_names"]:
                                    state["required_span_names"].append(name)
                for row in state["records"]:
                    row.update({f"sample_{k}": v for k, v in measurements.items()})
                cells = {
                    cell_key(r): {k: r[k] for k in CELL_FIELDS}
                    for r in state["records"]
                }
                execution = {
                    "execution_success": failure is None,
                    "n_records_completed": len(cells),
                    "no_answer_count": sum(r["no_answer"] for r in state["records"]),
                    "invalid_count": sum(not r["valid"] for r in state["records"]),
                }
                root.set_outputs(
                    state["outputs"]
                    | {"cells": cells, "runtime": measurements, "execution": execution}
                )
                for k, v in execution.items():
                    root.set_attribute(k, v)
                tracker.update_current_trace(
                    tags=execution
                    | {
                        "peak_vram_gib": measurements.get("peak_vram_bytes", 0) / 2**30,
                        "peak_reserved_vram_gib": measurements.get(
                            "peak_vram_reserved_bytes", 0
                        )
                        / 2**30,
                    },
                    response_preview=failure
                    or " | ".join(
                        f"{key}: {r['prediction']} ({'PASS' if r['correct'] else 'FAIL'}; valid={r['valid']})"
                        for key, r in cells.items()
                    )
                    or f"{phase}: completed",
                )
    finally:
        if state is not None and state["trace_id"]:
            names = assessments(tracker, state, item, measurements, failure)
            manifest.append(
                {
                    "trace_id": state["trace_id"],
                    "sample_id": sample_id,
                    "sample_index": index,
                    "upstream_steps": step,
                    "phase": phase,
                    "required_assessments": names,
                    "required_span_names": state["required_span_names"],
                    "execution_success": failure is None,
                }
            )


def record_agents(tracker, agents, context_id, source_trace_id, prefix="upstream"):
    """Show saved latent-agent snapshots without claiming additional inference."""
    names = []
    for index, agent in enumerate(agents):
        name = f"{prefix}_{index}_{agent.get('role', 'agent')}"
        names.append(name)
        with tracker.start_span(
            name,
            "AGENT",
            {
                "context_id": context_id,
                "source_trace_id": source_trace_id,
                "recorded_agent_snapshot": True,
                **agent,
            },
        ) as span:
            span.set_attribute("recorded_agent_snapshot", True)
            span.set_outputs(agent)
            span.set_attribute(
                "recorded_prompt_tokens", len(agent.get("input_ids", []))
            )
    return names


def record_budget(tracker, row):
    name = f"{row['handoff_condition']}_evaluate_r{row['receiver_budget']}"
    row["evaluation_span_name"] = name
    with tracker.start_span(
        name,
        "EVALUATOR",
        {
            "receiver_budget": row["receiver_budget"],
            "sample_id": row["sample_id"],
            "handoff_condition": row["handoff_condition"],
            "gold": row["gold"],
            "receiver_trace_id": row["receiver_trace_id"],
            "answer_policy": "explicit_complete_final_answer",
            "generated_token_ids": row["generated_token_ids"],
        },
    ) as span:
        span.set_outputs(row)
        span.set_attribute(
            "evaluation_status",
            "pathological"
            if not row["valid"]
            else "no_answer"
            if row["no_answer"]
            else "correct"
            if row["correct"]
            else "incorrect",
        )

    return name
