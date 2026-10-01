"""Bounded free-endpoint recovery and capped prefix verification."""

from contextlib import nullcontext

from transformers import set_seed

from ...domain.ports.tracking_port import DummySpan
from .attempt import finish_attempt, generate_attempt
from .inference import generate_receiver


def termination(model, ids, cap):
    config = getattr(getattr(model, "model", None), "generation_config", None)
    eos = getattr(config, "eos_token_id", None)
    if eos is None:
        eos = getattr(model.tokenizer, "eos_token_id", None)
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    if ids and ids[-1] in eos:
        return "eos"
    return "generation_stop" if len(ids) < cap else "cap"


def collect_trajectory(
    method,
    item,
    context,
    args,
    tracker=None,
    runtime=None,
    trace_inputs=None,
):
    budgets = [b for b in args.receiver_budgets if b != "free"]
    cap, previous, attempts = args.max_new_tokens, None, []
    generation_scope = (
        tracker.start_span(
            "free_generation",
            "CHAIN",
            {"initial_cap": cap, "retry_ceiling": args.free_max_new_tokens},
        )
        if tracker
        else nullcontext(DummySpan())
    )
    with generation_scope as generation_span:
        while True:
            set_seed(args.seed)
            ids, metadata = generate_attempt(
                generate_receiver,
                method,
                item,
                context,
                cap,
                budgets,
                "free",
                len(attempts),
                tracker,
                runtime,
                trace_inputs,
                termination,
            )
            if previous is not None and ids[: len(previous)] != previous:
                raise ValueError(
                    "Expanded free trajectory differs from initial greedy prefix"
                )
            reason = termination(method.model, ids, cap)
            attempts.append(finish_attempt(metadata, ids, cap, reason, method))
            if reason != "cap" or cap >= args.free_max_new_tokens:
                break
            previous = ids
            cap = min(cap * 2, args.free_max_new_tokens)
        generation_span.set_outputs(
            {
                "termination_reason": reason,
                "attempt_count": len(attempts),
                "final_cap": cap,
            }
        )
    natural = reason != "cap"
    verification_attempts = []
    if args.verify_prefix:
        verification_scope = (
            tracker.start_span("prefix_verification", "CHAIN", {"budgets": budgets})
            if tracker
            else nullcontext(DummySpan())
        )
        with verification_scope as verification_span:
            for budget in budgets:
                set_seed(args.seed)
                capped, capped_metadata = generate_attempt(
                    generate_receiver,
                    method,
                    item,
                    context,
                    budget,
                    (),
                    "prefix_verification",
                    len(verification_attempts),
                    tracker,
                    runtime,
                    trace_inputs,
                    termination,
                )
                verification_attempts.append(
                    finish_attempt(
                        capped_metadata,
                        capped,
                        budget,
                        termination(method.model, capped, budget),
                        method,
                    )
                )
            verification_span.set_outputs(
                {"capped_generations": len(verification_attempts)}
            )
    return {
        "token_ids": ids,
        "metadata": metadata,
        "prefix_verification_requested": args.verify_prefix,
        "free_attempts": attempts,
        "prefix_verification_attempts": verification_attempts,
        "free_final_cap": cap,
        "free_naturally_terminated": natural,
        "pathological": not natural,
        "free_initial_cap_reached": attempts[0]["termination_reason"] == "cap",
    }
