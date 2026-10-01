"""Sample-level answer, receiver cost, and donor diagnostics."""

from .answers import verify_prefix_evaluation
from .inference import evaluate_prefix


def trajectory_records(
    model,
    evaluator,
    item,
    args,
    trajectory,
    identity,
    recipient_meta,
    donor_meta,
    donor_trace,
):
    ids, metadata = trajectory["token_ids"], trajectory["metadata"]
    recipient_length = recipient_meta["handoff_positions"]
    donor_length = donor_meta.get("handoff_positions", 0)
    rows = []
    for budget in args.receiver_budgets:
        row = evaluate_prefix(
            model.tokenizer,
            evaluator,
            item,
            ids,
            budget,
            trajectory["free_naturally_terminated"],
            metadata["receiver_thinking_open"],
        )
        row.update(metadata)
        row.update(identity)
        row.update(
            question=item["question"],
            gold=item.get("gold", ""),
            reference_solution=item.get("solution"),
            model=args.model_name,
            seed=args.seed,
            recipient_sequence_length=recipient_length,
            donor_sequence_length=donor_length,
            recipient_upstream_metadata=recipient_meta,
            upstream_metadata=donor_meta,
            upstream_agents=donor_trace,
            free_generated_tokens=len(ids),
            free_cap_reached=trajectory["pathological"],
            valid=budget != "free" or not trajectory["pathological"],
            receiver_prompt=metadata["receiver_prompt"],
            receiver_input_ids=metadata["receiver_input_ids"],
            receiver_thinking_open=metadata["receiver_thinking_open"],
            receiver_budget_latency_sec=metadata["receiver_budget_latency_sec"],
            **{
                k: v
                for k, v in trajectory.items()
                if k not in {"token_ids", "metadata"}
            },
        )
        row["execution_success"] = True
        row["parse_failure"] = row["no_answer"]
        row["empty_output"] = not bool(row["raw_receiver_output"].strip())
        verify_prefix_evaluation(row, evaluator)
        rows.append(row)
    return rows
