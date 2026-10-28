"""Build the acquisition sender KV cache and retain its actual prompt inputs."""

import time
from typing import Any

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    move_past_kv,
    retain_past_kv_prefix,
    truncate_past_kv,
)
from ...domain.services.prompts import build_sender_messages
from ...infrastructure.models.model_wrapper import ModelWrapper
from .sampling import SecretDigitSample, SenderCacheBundle


def build_sender_cache(
    model: ModelWrapper, args: Any, sample: SecretDigitSample
) -> SenderCacheBundle:
    """Run sender agent and construct full, prompt-only, and latent-only KV caches."""
    messages = build_sender_messages(sample.digit)
    prompts, ids, mask, tokens = model.prepare_chat_batch(
        [messages], add_generation_prompt=True
    )
    started = time.perf_counter()
    full = model.generate_latent_batch(
        ids, attention_mask=mask, latent_steps=args.latent_steps
    )
    latency = time.perf_counter() - started
    prompt_len = int(mask.sum().item())
    expected = prompt_len + args.latent_steps
    if get_past_kv_sequence_length(full) != expected:
        raise RuntimeError(
            f"sender cache length mismatch: expected {expected}, got {get_past_kv_sequence_length(full)}"
        )
    full = move_past_kv(full, "cpu")
    latent = truncate_past_kv(clone_past_kv(full), args.latent_steps)
    prompt = retain_past_kv_prefix(clone_past_kv(full), prompt_len)
    if get_past_kv_sequence_length(latent) != args.latent_steps:
        raise RuntimeError("latent-only cache length mismatch")
    if get_past_kv_sequence_length(prompt) != prompt_len:
        raise RuntimeError("prompt-only cache length mismatch")
    return SenderCacheBundle(
        full=full,
        prompt_only=prompt,
        latent_only=latent,
        prompt_len=prompt_len,
        full_len=expected,
        build_latency_sec=latency,
        prompt_input={
            "messages": messages,
            "prompt": prompts[0],
            "input_ids": ids[0].cpu().tolist(),
            "attention_mask": mask[0].cpu().tolist(),
            "tokens": tokens[0],
            "add_generation_prompt": True,
        },
    )
