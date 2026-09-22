"""Forward execution passes and cache construction for receiver acquisition."""

import contextlib
import time
from typing import Any

import torch

from ...domain.models import ReceiverAcquisitionRecord
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...domain.services.kv_cache import (
    clone_past_kv,
    estimate_past_kv_bytes,
    get_past_kv_dtype,
    get_past_kv_num_layers,
    get_past_kv_sequence_length,
    move_past_kv,
    retain_past_kv_prefix,
    truncate_past_kv,
)
from ...domain.services.prompts import build_sender_messages
from ...infrastructure.models.model_wrapper import ModelWrapper
from .sampling import SecretDigitSample, SenderCacheBundle
from .scoring import score_digit_logits


def build_sender_cache(
    model: ModelWrapper, args: Any, sample: SecretDigitSample
) -> SenderCacheBundle:
    """Run sender agent and construct full, prompt-only, and latent-only KV caches."""
    _, ids, mask, _ = model.prepare_chat_batch(
        [build_sender_messages(sample.digit)], add_generation_prompt=True
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
    )


def forward_receiver_observation(
    model: ModelWrapper,
    args: Any,
    target: SecretDigitSample,
    source: SecretDigitSample | None,
    mode: str,
    condition: str,
    cache: Any,
    receiver_ids: torch.Tensor,
    receiver_mask: torch.Tensor,
    mapping: dict[str, Any],
    original_full_seq_len: int,
    tracker_port: ExperimentTrackerPort | None,
) -> tuple[ReceiverAcquisitionRecord, Any]:
    """Execute a single receiver forward observation under a specified cache condition."""
    span_name = (
        f"receiver_forward_{condition}"
        if condition == "drop"
        else f"receiver_forward_{mode}_{condition}"
    )
    span_ctx = (
        tracker_port.start_span(
            span_name,
            "LLM",
            {
                "target_digit": target.digit,
                "source_digit": source.digit if source else None,
                "context_mode": mode,
                "condition": condition,
            },
        )
        if tracker_port
        else contextlib.nullcontext(None)
    )
    started = time.perf_counter()
    error = None
    scored: dict[str, Any] = {}
    state_hidden = None
    cache_for_forward = (
        move_past_kv(clone_past_kv(cache), model.device) if cache is not None else None
    )
    cache_seq_len = get_past_kv_sequence_length(cache)
    if condition == "drop":
        receiver_position_start = 0
        retained_start = None
        retained_end = None
        retained_tail_len = 0
    elif condition == "drop_position_matched":
        receiver_position_start = original_full_seq_len
        retained_start = None
        retained_end = None
        retained_tail_len = 0
    elif mode == "prompt_only":
        receiver_position_start = cache_seq_len
        retained_start = 0
        retained_end = cache_seq_len
        retained_tail_len = cache_seq_len
    elif mode == "latent_only_position_fixed":
        receiver_position_start = original_full_seq_len
        retained_start = original_full_seq_len - cache_seq_len
        retained_end = original_full_seq_len
        retained_tail_len = cache_seq_len
    elif mode in {"latent_only", "latent_only_compact_debug"}:
        receiver_position_start = cache_seq_len
        retained_start = original_full_seq_len - cache_seq_len
        retained_end = original_full_seq_len
        retained_tail_len = cache_seq_len
    else:
        receiver_position_start = original_full_seq_len
        retained_start = 0
        retained_end = original_full_seq_len
        retained_tail_len = original_full_seq_len

    with span_ctx as span:
        try:
            state = model.forward_next_token_batch(
                receiver_ids,
                receiver_mask,
                past_key_values=cache_for_forward,
                output_hidden_states=args.save_hidden_states,
                position_start=receiver_position_start,
            )
            state_hidden = state.hidden_states
            scored = score_digit_logits(
                state.logits, mapping, source.digit if source else None, target.digit
            )
            if span is not None:
                span.set_outputs(
                    scored
                    | {
                        "cache_sequence_length": cache_seq_len,
                        "latency_sec": time.perf_counter() - started,
                    }
                )
                span.set_token_usage(int(receiver_mask.sum().item()), 0)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if span is not None:
                span.set_status("ERROR", error)
    latency = time.perf_counter() - started
    empty = {
        "candidate_probabilities": {},
        "candidate_log_probabilities": {},
        "candidate_mass": 0.0,
        "predicted_digit": None,
        "source_probability": None,
        "source_log_probability": None,
        "source_rank": None,
        "target_probability": None,
        "target_log_probability": None,
        "target_rank": None,
        "source_margin": None,
        "content_follow_correct": None,
        "target_retained": None,
    }
    values = scored or empty
    artifact_key = (
        f"{target.sample_id}/{mode}/{condition}"
        if args.save_hidden_states and not error
        else None
    )
    return (
        ReceiverAcquisitionRecord(
            sample_id=target.sample_id,
            sample_index=target.sample_index,
            sample_key=target.sample_key,
            target_digit=target.digit,
            source_sample_id=source.sample_id if source else None,
            source_sample_index=source.sample_index if source else None,
            source_sample_key=source.sample_key if source else None,
            source_digit=source.digit if source else None,
            context_mode=mode,
            condition=condition,
            **values,
            cache_present=cache is not None,
            cache_sequence_length=cache_seq_len,
            cache_bytes=estimate_past_kv_bytes(cache),
            num_layers=get_past_kv_num_layers(cache),
            cache_dtype=get_past_kv_dtype(cache),
            original_full_seq_len=original_full_seq_len,
            retained_tail_len=retained_tail_len,
            retained_tail_start_position=retained_start,
            retained_original_position_end=retained_end,
            receiver_position_start=receiver_position_start,
            receiver_prompt_tokens=int(receiver_mask.sum().item()),
            latency_sec=latency,
            error=error,
            hidden_state_artifact_key=artifact_key,
            seed=args.seed,
            model=args.model_name,
            task=args.task,
            latent_steps=args.latent_steps,
        ),
        state_hidden,
    )
