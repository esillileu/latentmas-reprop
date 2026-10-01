"""Full-cache receiver trajectories and exact budget-prefix evaluation."""

import torch

from ...domain.services.kv_cache import clone_past_kv, move_past_kv
from ...infrastructure.models.text_generation import generate_token_ids_batch


def length_matched_donors(lengths):
    """Rotate neighboring lengths, giving a deterministic bijective derangement."""
    if len(lengths) < 2:
        raise ValueError("Mismatched handoff requires at least two samples")
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    donors = [None] * len(order)
    for i, recipient in enumerate(order):
        donors[recipient] = order[(i + 1) % len(order)]
    return donors


@torch.no_grad()
def generate_receiver(method, item, context, limit, report_progress=False):
    model = method.model
    prompts, _, _, _ = model.prepare_chat_batch(
        method._build_messages("judger", [item]), add_generation_prompt=True
    )
    wrapped, ids, mask, _ = method._prepare_tokens(prompts, model.device)
    token_ids, _ = generate_token_ids_batch(
        model.model,
        model.tokenizer,
        model.device,
        ids,
        mask,
        max_new_tokens=limit,
        temperature=0.0,
        top_p=1.0,
        past_key_values=move_past_kv(clone_past_kv(context), model.device),
        report_progress=report_progress,
    )
    return token_ids[0], {
        "receiver_prompt": wrapped[0],
        "receiver_input_ids": ids[0].tolist(),
    }


def evaluate_prefix(tokenizer, evaluator, item, token_ids, budget):
    ids = token_ids if budget == "free" else token_ids[:budget]
    text = tokenizer.decode(ids, skip_special_tokens=True).strip()
    prediction, correct, error = evaluator.evaluate("gsm8k", text, item.get("gold", ""))
    return {
        "receiver_budget": budget,
        "generated_token_ids": ids,
        "generated_tokens": len(ids),
        "raw_receiver_output": text,
        "prediction": prediction,
        "correct": bool(correct),
        "error_msg": error,
    }
