"""Collect paired handoff cells with sample and inference tracing."""

from contextlib import nullcontext

from transformers import set_seed

from ...domain.models import compute_sample_key
from ...domain.ports.tracking_port import DummySpan
from ...domain.services.latent_mas import LatentMASMethod
from ..common.reporter import write_jsonl
from .diagnostics import execution_counts, inference_costs
from .inference import build_upstream, decode_receiver
from .metrics import sample_measurement_tags


def collect_sample(
    model,
    args,
    item,
    index,
    config,
    records,
    records_path,
    evaluator_port,
    tracker_port,
    runtime,
):
    sample_id = compute_sample_key(args.task, args.split, item["question"])
    root_context = (
        tracker_port.start_sample_trace(
            "receiver_reasoning_sample",
            {
                "sample_id": sample_id,
                "sample_index": index,
                "question": item["question"],
                "gold": item.get("gold", ""),
            },
            tags={
                "sample_id": sample_id,
                "sample_index": index,
                "model": args.model_name,
                "task": args.task,
                "seed": args.seed,
                "handoff_mode": args.handoff_mode,
            },
            request_preview=f"[{index}] {item['question'][:180]}",
        )
        if tracker_port
        else nullcontext(DummySpan())
    )
    first = len(records)
    with root_context as root:
        sample_measurements = {}
        try:
            with runtime.measure(root) as sample_measurements:
                for step in [0, *args.upstream_steps]:
                    method = LatentMASMethod(
                        model,
                        latent_steps=step,
                        judger_max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        generate_bs=1,
                        args=args,
                        evaluator=evaluator_port,
                    )
                    set_seed(args.seed)
                    context_id = f"{sample_id}:{step}"
                    context, traces, measurements = build_upstream(
                        method,
                        item,
                        step,
                        args.handoff_positions,
                        context_id,
                        tracker_port,
                        runtime,
                    )
                    length = measurements["handoff_positions"]
                    for mode in ("answer_only", "free"):
                        set_seed(args.seed)
                        limit = (
                            args.answer_only_max_new_tokens
                            if mode == "answer_only"
                            else args.max_new_tokens
                        )
                        print(
                            f"[receiver] sample={index} upstream_steps={step} "
                            f"mode={mode} budget={limit}",
                            flush=True,
                        )
                        result, diagnostics = decode_receiver(
                            method,
                            item,
                            context,
                            traces,
                            mode,
                            limit,
                            step,
                            context_id,
                            measurements,
                            tracker_port,
                            runtime,
                        )
                        records.append(
                            {
                                "sample_id": sample_id,
                                "sample_index": index,
                                "question": item["question"],
                                "gold": item.get("gold", ""),
                                "upstream_latent_steps": step,
                                "receiver_mode": mode,
                                "handoff_condition": "latent" if step else "no_handoff",
                                **measurements,
                                **diagnostics,
                                "trace_id": root.trace_id,
                                "reference_solution": item.get("solution"),
                                "receiver_max_new_tokens": limit,
                                "context_id": f"{sample_id}:{step}",
                                "upstream_context_sequence_length": length,
                                "prediction": result["prediction"],
                                "raw_receiver_output": result["raw_prediction"],
                                "correct": result["correct"],
                                "error_msg": result.get("error_msg"),
                                "model": args.model_name,
                                "seed": args.seed,
                                "config": config,
                                "agents": result["agents"],
                            }
                        )
                        write_jsonl(records_path, records)
                    del context
        finally:
            sample_records = records[first:]
            outputs = {
                f"{r['upstream_latent_steps']}_{r['receiver_mode']}": {
                    "prediction": r["prediction"],
                    "correct": r["correct"],
                    "token_limit_reached": r["receiver_token_limit_reached"],
                    "parse_failure": r["parse_failure"],
                    "execution_success": r["execution_success"],
                    "receiver_latency_sec": r["receiver_latency_sec"],
                }
                for r in sample_records
            }
            sample_measurements.update(inference_costs(sample_records))
            counts = execution_counts(
                sample_records, 2 * (1 + len(args.upstream_steps))
            )
            for name, value in {**sample_measurements, **counts}.items():
                root.set_attribute(name, value)
            for row in sample_records:
                row.update({f"sample_{k}": v for k, v in sample_measurements.items()})
            write_jsonl(records_path, records)
            root.set_outputs(
                {"cells": outputs, "runtime": sample_measurements, "execution": counts}
            )
        if tracker_port:
            tracker_port.update_current_trace(
                tags=sample_measurement_tags(sample_measurements),
                response_preview=" | ".join(
                    f"{key}: {str(value['prediction'])[:60]} ({'PASS' if value['correct'] else 'FAIL'})"
                    for key, value in outputs.items()
                ),
            )
        trace_id = root.trace_id
    if tracker_port and trace_id:
        tracker_port.log_expectation(trace_id, "expected_answer", item.get("gold", ""))
        if item.get("solution"):
            tracker_port.log_expectation(
                trace_id, "reference_solution", str(item["solution"])
            )
        errors = [r["error_msg"] for r in sample_records if r.get("error_msg")]
        tracker_port.log_feedback(
            trace_id,
            "execution_success",
            not errors,
            rationale="; ".join(errors) or None,
        )
        for key, value in outputs.items():
            tracker_port.log_feedback(trace_id, f"{key}_correct", value["correct"])
            tracker_port.log_feedback(
                trace_id, f"{key}_truncated", value["token_limit_reached"]
            )
