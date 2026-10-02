"""Condition traces with compact results and correctness-only assessments."""

import traceback
from contextlib import contextmanager

from ..common.reporter import build_run_stem


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


def evaluation_result(row):
    status = (
        "token_limit"
        if not row["valid"]
        else "no_answer"
        if row["no_answer"]
        else "correct"
        if row["correct"]
        else "incorrect"
    )
    result = {
        "budget": row["receiver_budget"],
        "status": status,
        "correct": row["correct"],
        "response": row["raw_receiver_output"],
    }
    if row["prediction"] is not None:
        result["answer"] = row["prediction"]
    return result


@contextmanager
def sample_trace(
    tracker, runtime, args, config, item, index, sample_id, step, condition, manifest
):
    settings = {
        "model": args.model_name,
        "upstream_steps": step,
        "handoff_condition": condition,
        "receiver_budgets": args.receiver_budgets,
        "max_new_tokens": args.max_new_tokens,
        "free_max_new_tokens": args.free_max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": args.seed,
    }
    tags = {
        "experiment_type": "receiver_compute_preflight",
        "task": args.task,
        "split": args.split,
        "sample_id": sample_id,
        "sample_index": index,
        "handoff_mode": "full",
        "receiver_mode": "free",
        **{
            k: config[k]
            for k in (
                "git_commit",
                "version_tag",
                "run_name",
                "run_kind",
                "run_label",
                "run_id",
            )
        },
        **{
            k: settings[k]
            for k in ("model", "upstream_steps", "handoff_condition", "seed")
        },
    }
    inputs = {
        "question": item["question"],
        "expected_answer": item.get("gold", ""),
        "settings": settings,
    }
    if item.get("solution"):
        inputs["reference_solution"] = item["solution"]
    root_name = f"{args.model_name.split('/')[-1]}_sample_{index}_u{step}_{condition}"
    state, measurements, failure = None, {}, None
    try:
        with tracker.start_sample_trace(
            root_name,
            inputs,
            tags=tags,
            request_preview=f"[{index}] U={step} {condition}: {item['question'][:160]}",
        ) as root:
            state = {
                "trace_id": root.trace_id,
                "records": [],
                "required_span_names": [root_name],
            }
            try:
                with runtime.measure(root) as measurements:
                    yield state
            except (Exception, KeyboardInterrupt) as exc:
                failure = f"{type(exc).__name__}: {exc}"
                root.set_status("ERROR", failure)
                root.set_attribute("failure_traceback", traceback.format_exc())
                raise
            finally:
                for row in state["records"]:
                    row.update({f"sample_{k}": v for k, v in measurements.items()})
                    for field in ("free_attempts", "prefix_verification_attempts"):
                        for attempt in row[field]:
                            name = attempt["receiver_span_name"]
                            if name not in state["required_span_names"]:
                                state["required_span_names"].append(name)
                evaluations = [evaluation_result(row) for row in state["records"]]
                free = next(
                    (
                        row
                        for row in state["records"]
                        if row["receiver_budget"] == "free"
                    ),
                    None,
                )
                outputs = {"evaluations": evaluations}
                if free is not None:
                    outputs = {
                        "response": free["raw_receiver_output"],
                        "status": evaluation_result(free)["status"],
                        **outputs,
                    }
                    for key in (
                        "free_final_cap",
                        "free_naturally_terminated",
                        "donor_id",
                        "donor_sequence_length",
                        "upstream_trace_id",
                    ):
                        if free.get(key) is not None:
                            root.set_attribute(key, free[key])
                if failure:
                    outputs.update(status="execution_error", error=failure)
                root.set_outputs(outputs)
                root.set_attribute("execution_success", failure is None)
                root.set_attribute("n_records_completed", len(evaluations))
                preview = failure or " | ".join(
                    f"R={e['budget']}: {e.get('answer', e['status'])}"
                    for e in evaluations
                )
                tracker.update_current_trace(
                    tags={
                        "execution_success": failure is None,
                        "result_status": outputs.get("status", "execution_error"),
                        "peak_vram_gib": measurements.get("peak_vram_bytes", 0) / 2**30,
                    },
                    response_preview=preview,
                )
    finally:
        if state is not None and state["trace_id"]:
            names = []
            for row in state["records"]:
                name = f"correct_{'free' if row['receiver_budget'] == 'free' else 'r' + str(row['receiver_budget'])}"
                tracker.log_feedback(
                    state["trace_id"],
                    name,
                    row["correct"],
                    rationale=f"R={row['receiver_budget']}; {evaluation_result(row)['status']}; expected={item.get('gold', '')}",
                )
                names.append(name)
            manifest.append(
                {
                    "trace_id": state["trace_id"],
                    "sample_id": sample_id,
                    "sample_index": index,
                    "upstream_steps": step,
                    "handoff_condition": condition,
                    "required_assessments": names,
                    "required_span_names": state["required_span_names"],
                    "execution_success": failure is None,
                }
            )


def record_budgets(tracker, rows):
    names = ["budget_evaluation"]
    with tracker.start_span(
        "budget_evaluation", "CHAIN", {"budgets": [r["receiver_budget"] for r in rows]}
    ) as evaluation_span:
        for row in rows:
            name = f"evaluate_{'free' if row['receiver_budget'] == 'free' else 'r' + str(row['receiver_budget'])}"
            names.append(name)
            row["evaluation_span_name"] = name
            with tracker.start_span(
                name,
                "EVALUATOR",
                {"budget": row["receiver_budget"], "expected_answer": row["gold"]},
            ) as span:
                span.set_outputs(evaluation_result(row))
                for key in (
                    "correct",
                    "no_answer",
                    "valid",
                    "final_answer_complete",
                    "generated_tokens",
                ):
                    span.set_attribute(key, row[key])
        evaluation_span.set_outputs({"budgets_evaluated": len(rows)})
    return names
