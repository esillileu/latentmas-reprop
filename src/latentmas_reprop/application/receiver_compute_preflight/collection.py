"""Collect sample/condition traces while reusing paired KV and baseline inference."""

from transformers import set_seed

from ...domain.services.kv_cache import move_past_kv
from ...domain.services.latent_mas import LatentMASMethod
from ..common.reporter import write_jsonl
from ..receiver_reasoning.inference import build_upstream
from .inference import length_matched_donors
from .records import trajectory_records
from .tracing import record_budgets, sample_trace
from .trajectory import collect_trajectory


def collect_condition(
    method,
    args,
    item,
    index,
    sample_id,
    step,
    condition,
    config,
    tracker,
    evaluator,
    runtime,
    manifest,
    donor=None,
    recipient_meta=None,
):
    """One trace owns one sample, upstream level, and handoff condition."""
    with sample_trace(
        tracker,
        runtime,
        args,
        config,
        item,
        index,
        sample_id,
        step,
        condition,
        manifest,
    ) as state:
        if condition == "matched":
            set_seed(args.seed)
            context, agents, metadata = build_upstream(
                method,
                item,
                step,
                None,
                f"{sample_id}:{step}",
                tracker,
                runtime,
            )
            agents = agents[0]
            donor_id, source_trace_id = sample_id, state["trace_id"]
            recipient_meta = metadata
            state["required_span_names"].append(f"build_upstream_{step}")
        elif donor is not None:
            donor_id, context, agents, metadata, source_trace_id = donor
            with tracker.start_span(
                "reuse_upstream",
                "CHAIN",
                {
                    "donor_id": donor_id,
                    "source_trace_id": source_trace_id,
                    "context_id": f"{donor_id}:{step}",
                },
            ) as span:
                for key, value in metadata.items():
                    if (
                        key.startswith("cache_") or key == "handoff_positions"
                    ) and value is not None:
                        span.set_attribute(key, value)
                span.set_outputs(
                    {"cache_reused": True, "source_trace_id": source_trace_id}
                )
            state["required_span_names"].append("reuse_upstream")
        else:
            donor_id, context, agents, metadata, source_trace_id = (
                None,
                None,
                [],
                {},
                None,
            )
            recipient_meta = {"handoff_positions": 0}
        identity = {
            "sample_id": sample_id,
            "sample_index": index,
            "upstream_steps": step,
            "handoff_condition": condition,
            "donor_id": donor_id,
            "context_id": f"{donor_id}:{step}" if donor_id else None,
            "upstream_trace_id": source_trace_id,
        }
        trace_inputs = {k: v for k, v in identity.items() if v is not None}
        trace_inputs["model"] = args.model_name
        with tracker.start_span("receiver", "CHAIN", trace_inputs) as span:
            trajectory = collect_trajectory(
                method,
                item,
                context,
                args,
                tracker,
                runtime,
                trace_inputs,
            )
            span.set_outputs(
                {
                    "generated_tokens": len(trajectory["token_ids"]),
                    "termination_reason": trajectory["free_attempts"][-1][
                        "termination_reason"
                    ],
                }
            )
        state["required_span_names"].extend(["receiver", "free_generation"])
        if args.verify_prefix:
            state["required_span_names"].append("prefix_verification")
        rows = trajectory_records(
            method.model,
            evaluator,
            item,
            args,
            trajectory,
            identity,
            recipient_meta,
            metadata,
            agents,
        )
        for row in rows:
            row["trace_id"] = state["trace_id"]
        state["records"].extend(rows)
        state["required_span_names"].extend(record_budgets(tracker, rows))
    return (
        sample_id,
        move_past_kv(context, "cpu"),
        agents,
        metadata,
        state["trace_id"],
    ), rows


def collect_samples(
    model,
    args,
    items,
    ids,
    config,
    tracker,
    evaluator,
    runtime,
    records,
    manifest,
    records_path,
):
    method = LatentMASMethod(
        model,
        latent_steps=0,
        args=args,
        evaluator=evaluator,
        temperature=0.0,
        top_p=1.0,
        generate_bs=1,
    )
    for index, item in enumerate(items):
        _, rows = collect_condition(
            method,
            args,
            item,
            index,
            ids[index],
            0,
            "no_handoff",
            config,
            tracker,
            evaluator,
            runtime,
            manifest,
        )
        for step in args.upstream_steps:
            records.extend(
                row | {"upstream_steps": step, "baseline_reused": True} for row in rows
            )
        write_jsonl(records_path, records)
    for step in args.upstream_steps:
        method = LatentMASMethod(
            model,
            latent_steps=step,
            args=args,
            evaluator=evaluator,
            temperature=0.0,
            top_p=1.0,
            generate_bs=1,
        )
        donors = []
        for index, item in enumerate(items):
            donor, rows = collect_condition(
                method,
                args,
                item,
                index,
                ids[index],
                step,
                "matched",
                config,
                tracker,
                evaluator,
                runtime,
                manifest,
            )
            donors.append(donor)
            records.extend(rows)
            # U-independent baseline inference and evaluations keep their original trace.
            for row in records:
                if (
                    row["sample_id"] == ids[index]
                    and row["upstream_steps"] == step
                    and row["handoff_condition"] == "no_handoff"
                ):
                    row.update(
                        recipient_sequence_length=donor[3]["handoff_positions"],
                        recipient_upstream_metadata=donor[3],
                    )
            write_jsonl(records_path, records)
        sources = length_matched_donors(
            [donor[3]["handoff_positions"] for donor in donors]
        )
        for index, item in enumerate(items):
            _, rows = collect_condition(
                method,
                args,
                item,
                index,
                ids[index],
                step,
                "mismatched",
                config,
                tracker,
                evaluator,
                runtime,
                manifest,
                donor=donors[sources[index]],
                recipient_meta=donors[index][3],
            )
            records.extend(rows)
            write_jsonl(records_path, records)
            print(
                f"[compute preflight] U={step} sample={index + 1}/{len(items)}",
                flush=True,
            )
