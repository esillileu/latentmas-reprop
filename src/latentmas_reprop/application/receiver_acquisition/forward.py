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
)
from ...infrastructure.models.model_wrapper import ModelWrapper
from .sampling import SecretDigitSample
from .scoring import score_digit_logits


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
    raw_logits: dict[str, torch.Tensor] | None = None,
) -> tuple[ReceiverAcquisitionRecord, Any]:
    """Execute a single receiver forward observation under a specified cache condition."""
    active_ids = receiver_ids[0][receiver_mask[0].bool()].cpu().tolist()
    receiver_input = {
        "messages": mapping["messages"],
        "prompt": mapping["prompt"],
        "input_ids": receiver_ids[0].cpu().tolist(),
        "attention_mask": receiver_mask[0].cpu().tolist(),
        "tokens": model.tokenizer.convert_ids_to_tokens(active_ids),
        "chat_template_kwargs": mapping["chat_template_kwargs"],
        "answer_prefix": mapping["answer_prefix"],
    }
    raw_output = {}
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
                "latent_steps": args.latent_steps,
                "original_full_seq_len": original_full_seq_len,
                **receiver_input,
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
    elif mode == "latent_only":
        receiver_position_start = original_full_seq_len
        retained_start = original_full_seq_len - cache_seq_len
        retained_end = original_full_seq_len
        retained_tail_len = cache_seq_len
    elif mode == "latent_only_compact_debug":
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
            logits_key = f"{mode}/{condition}/step_{args.latent_steps}"
            if raw_logits is not None:
                raw_logits[logits_key] = state.logits.detach().cpu().clone()
            scored = score_digit_logits(
                state.logits, mapping, source.digit if source else None, target.digit
            )
            top_values, top_ids = torch.topk(
                torch.log_softmax(state.logits[0].float(), dim=-1), k=5
            )
            scored["top_token_candidates"] = [
                {
                    "token_id": int(token_id),
                    "token": model.tokenizer.convert_ids_to_tokens(int(token_id)),
                    "text": model.tokenizer.decode([int(token_id)]),
                    "probability": float(value.exp().item()),
                }
                for value, token_id in zip(top_values, top_ids, strict=True)
            ]
            vocabulary_id = int(state.logits[0].argmax().item())
            digit = str(scored["predicted_digit"])
            digit_id = mapping["candidates"][digit]["contextual_token_id"]
            raw_output = {
                "type": "next_token_logits",
                "digit_argmax": {
                    "digit": int(digit),
                    "token_id": digit_id,
                    "text": model.tokenizer.decode([digit_id]),
                },
                "vocabulary_argmax": {
                    "token_id": vocabulary_id,
                    "text": model.tokenizer.decode([vocabulary_id]),
                    "logit": float(state.logits[0, vocabulary_id].item()),
                },
                "logits_shape": list(state.logits.shape),
                "logits_dtype": str(state.logits.dtype),
                "logits_artifact": f"receiver_outputs/{target.sample_id}.pt"
                if raw_logits is not None
                else None,
                "logits_key": logits_key if raw_logits is not None else None,
            }
            if span is not None:
                span.set_outputs(
                    scored
                    | {
                        "raw_output": raw_output,
                        "cache_sequence_length": cache_seq_len,
                        "original_full_seq_len": original_full_seq_len,
                        "receiver_position_start": receiver_position_start,
                        "retained_tail_start_position": retained_start,
                        "retained_original_position_end": retained_end,
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
        "top_token_candidates": [],
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
            receiver_input=receiver_input,
            raw_output=raw_output,
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
