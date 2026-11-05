"""Continue through Critic, Refiner and Judger while retaining the full KV."""

import copy
from typing import Any

import torch

from ...domain.services.kv_cache import get_past_kv_sequence_length
from .conditions import sequential_prompts
from .decoder import forward_intervention_prefix, run_explicit_generation
from .determinism import deterministic_forward


@torch.no_grad()
def execute_sequential(
    runner: Any, item: dict, kv_snapshot: Any, steps: int
) -> tuple[torch.Tensor, dict]:
    model = runner.model
    kv = kv_snapshot
    stages = sequential_prompts(model, item, runner.args)
    blocks = []
    latent_runner = copy.copy(runner)
    latent_runner.args = copy.copy(runner.args)
    latent_runner.args.latent_steps = steps
    for stage in stages:
        ids = torch.tensor([stage["token_ids"]], dtype=torch.long)
        stage["cache_length_before"] = get_past_kv_sequence_length(kv)
        with deterministic_forward(torch.device(model.device)):
            kv, hidden = forward_intervention_prefix(
                model, kv, ids, torch.ones_like(ids)
            )
        count = steps if stage["role"] != "judger" else 0
        if count:
            kv, hidden, _ = latent_runner.compute_common_latent(
                dict(item, prompt_length=len(stage["token_ids"])),
                initial_kv=kv,
                initial_hiddens=hidden,
            )
        stage.update(
            latent_steps=count,
            cache_length_after=get_past_kv_sequence_length(kv),
            hidden_offset=sum(block.shape[0] for block in blocks),
            hidden_length=int(hidden.shape[0]),
        )
        blocks.append(hidden)

    ids, generated_hidden, reason, exhausted, text = run_explicit_generation(
        model, kv, blocks[-1][-1], runner.args, desc="Decoding · Judger"
    )
    generated_offset = sum(block.shape[0] for block in blocks) - 1
    if len(generated_hidden) > 1:
        blocks.append(torch.stack(generated_hidden[1:]))
    prediction, correct, error = runner.evaluator.evaluate(
        getattr(runner.args, "task", "gsm8k"), text, item.get("gold", "")
    )
    return torch.cat(blocks), {
        "receiver_role": "judger",
        "intervention_type": "sequential",
        "intervention_length": sum(len(stage["token_ids"]) for stage in stages),
        "stages": stages,
        "latent_steps_per_role": steps,
        "latent_steps_executed": steps * 3,
        "generated_hidden_offset": generated_offset,
        "generated_text": text,
        "generated_token_ids": ids,
        "generated_tokens": model.tokenizer.convert_ids_to_tokens(ids),
        "generated_length": len(ids),
        "finish_reason": reason,
        "budget_exhausted": exhausted,
        "prediction": prediction,
        "gold": item.get("gold", ""),
        "correct": correct,
        "error_msg": error,
    }
