"""Full-cache receiver trajectories and exact budget-prefix evaluation."""

import torch
from scipy.optimize import linear_sum_assignment

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    move_past_kv,
)
from ...infrastructure.models.text_generation import generate_token_ids_batch
from .answers import evaluate_final_answer
from .cost import ReceiverTiming


def length_matched_donors(lengths):
    """Minimize total absolute cache-length difference over all derangements."""
    if len(lengths) < 2:
        raise ValueError("Mismatched handoff requires at least two samples")
    costs = [
        [abs(a - b) if i != j else float("inf") for j, b in enumerate(lengths)]
        for i, a in enumerate(lengths)
    ]
    recipients, sources = linear_sum_assignment(costs)
    donors = [None] * len(lengths)
    for recipient, donor in zip(recipients, sources, strict=True):
        donors[int(recipient)] = int(donor)
    return donors


@torch.no_grad()
def generate_receiver(method, item, context, limit, budgets=(), report_progress=False):
    model = method.model
    prompts, _, _, _ = model.prepare_chat_batch(
        method._build_messages("judger", [item]), add_generation_prompt=True
    )
    wrapped, ids, mask, _ = method._prepare_tokens(prompts, model.device)
    cache = move_past_kv(clone_past_kv(context), model.device)
    timing = ReceiverTiming(ids.shape[-1], budgets, model.device)
    timing.start()
    token_ids, _ = generate_token_ids_batch(
        model.model,
        model.tokenizer,
        model.device,
        ids,
        mask,
        max_new_tokens=limit,
        temperature=0.0,
        top_p=1.0,
        past_key_values=cache,
        observer=timing,
        report_progress=report_progress,
    )
    latency = timing.elapsed()
    return token_ids[0], {
        "receiver_prompt": wrapped[0],
        "receiver_input_ids": ids[0].tolist(),
        "receiver_thinking_open": wrapped[0].rfind("<think>")
        > wrapped[0].rfind("</think>"),
        "receiver_cache_positions": get_past_kv_sequence_length(context),
        "receiver_prompt_tokens": int(mask.sum()),
        "receiver_latency_sec": latency,
        "receiver_budget_latency_sec": {
            str(k): v for k, v in timing.checkpoints.items()
        },
    }


def evaluate_prefix(
    tokenizer,
    evaluator,
    item,
    token_ids,
    budget,
    naturally_terminated=False,
    thinking_open=False,
):
    ids = token_ids if budget == "free" else token_ids[:budget]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    evaluation = evaluate_final_answer(
        evaluator,
        text,
        item.get("gold", ""),
        naturally_terminated and len(ids) == len(token_ids),
        thinking_open,
    )
    return {
        "receiver_budget": budget,
        "generated_token_ids": ids,
        "generated_tokens": len(ids),
        "raw_receiver_output": text,
        **evaluation,
    }
