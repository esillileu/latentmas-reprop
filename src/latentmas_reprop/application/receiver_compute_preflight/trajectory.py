"""Bounded free-endpoint recovery and capped prefix verification."""

from transformers import set_seed

from .attempt import finish_attempt, generate_attempt
from .inference import evaluate_prefix, generate_receiver


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
    evaluator,
    tracker=None,
    runtime=None,
    trace_inputs=None,
):
    budgets = [b for b in args.receiver_budgets if b != "free"]
    cap, previous, attempts = args.max_new_tokens, None, []
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
            evaluator,
            termination,
        )
        if previous is not None and ids[: len(previous)] != previous:
            raise ValueError(
                "Expanded free trajectory differs from initial greedy prefix"
            )
        reason = termination(method.model, ids, cap)
        attempts.append(
            finish_attempt(metadata, ids, cap, reason, method, item, evaluator)
        )
        if reason != "cap" or cap >= args.free_max_new_tokens:
            break
        previous = ids
        cap = min(cap * 2, args.free_max_new_tokens)
    natural = reason != "cap"
    verified = False
    verification_attempts = []
    if args.verify_prefix:
        prompt_fields = (
            "receiver_prompt",
            "receiver_input_ids",
            "receiver_thinking_open",
        )
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
                evaluator,
                termination,
            )
            verification_attempts.append(
                finish_attempt(
                    capped_metadata,
                    capped,
                    budget,
                    termination(method.model, capped, budget),
                    method,
                    item,
                    evaluator,
                )
            )
            if capped != ids[:budget] or any(
                capped_metadata[k] != metadata[k] for k in prompt_fields
            ):
                raise ValueError(
                    f"Greedy prefix differs from capped generation at R={budget}"
                )
            expected = evaluate_prefix(
                method.model.tokenizer,
                evaluator,
                item,
                ids,
                budget,
                natural,
                metadata["receiver_thinking_open"],
            )
            actual = evaluate_prefix(
                method.model.tokenizer,
                evaluator,
                item,
                capped,
                budget,
                termination(method.model, capped, budget) != "cap",
                metadata["receiver_thinking_open"],
            )
            if expected != actual:
                raise ValueError("Capped evaluation differs from prefix evaluation")
        verified = True
    return {
        "token_ids": ids,
        "metadata": metadata,
        "prefix_verified": verified,
        "free_attempts": attempts,
        "prefix_verification_attempts": verification_attempts,
        "free_final_cap": cap,
        "free_naturally_terminated": natural,
        "pathological": not natural,
        "free_initial_cap_reached": attempts[0]["termination_reason"] == "cap",
        "free_retry_count": len(attempts) - 1,
        "receiver_retry_latency_sec": sum(
            a["receiver_latency_sec"] for a in attempts[:-1]
        ),
    }
