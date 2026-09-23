"""Latent rollout and hidden state trajectory generation."""

from collections.abc import Callable
from typing import Any

import torch

from ...domain.ports.model_port import LatentRolloutState
from ...domain.services.kv_cache import get_past_kv_sequence_length


def _require_finite(tensor: torch.Tensor, stage: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite value detected during {stage}.")


def generate_latent_rollout(
    model: torch.nn.Module,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    latent_steps: int,
    past_key_values: Any = None,
    apply_realign_fn: Callable[[torch.Tensor, torch.nn.Module], torch.Tensor],
    source_model: torch.nn.Module | None = None,
) -> LatentRolloutState:
    """Execute autoregressive latent rollouts injecting hidden states as embeddings."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    if past_key_values is not None:
        past_len = get_past_kv_sequence_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]
    _require_finite(last_hidden, "latent prompt forward")
    hidden_steps: list[torch.Tensor] = []
    latent_steps_output: list[torch.Tensor] = []

    src_model = source_model or model
    for _ in range(latent_steps):
        hidden_steps.append(last_hidden.detach())
        latent_vec = apply_realign_fn(last_hidden, src_model)
        _require_finite(latent_vec, "latent realignment")
        latent_steps_output.append(latent_vec.detach())
        latent_embed = latent_vec.unsqueeze(1)

        past_len = get_past_kv_sequence_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long,
            device=device,
        )
        outputs = model(
            inputs_embeds=latent_embed,
            attention_mask=latent_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

    batch_size = input_ids.shape[0]
    hidden_size = last_hidden.shape[-1]
    empty = last_hidden.new_empty((batch_size, 0, hidden_size))
    return LatentRolloutState(
        past_key_values=past,
        hidden_pre_realign=torch.stack(hidden_steps, dim=1) if hidden_steps else empty,
        latent_post_realign=torch.stack(latent_steps_output, dim=1)
        if latent_steps_output
        else empty.clone(),
    )


def generate_latent_hidden_state_batch(
    hf_model: torch.nn.Module,
    hf_device: torch.device | str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    latent_steps: int,
    past_key_values: Any = None,
    apply_realign_fn: Callable[[torch.Tensor, torch.nn.Module], torch.Tensor],
) -> tuple[Any, torch.Tensor]:
    """Generate latent hidden state representations along with past KV."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
    device = torch.device(hf_device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    if past_key_values is not None:
        past_len = get_past_kv_sequence_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    outputs = hf_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    past = outputs.past_key_values
    last_hidden = outputs.hidden_states[-1][:, -1, :]

    curr_output_embedding = [outputs.hidden_states[0]]

    for _ in range(latent_steps):
        latent_vec = apply_realign_fn(last_hidden, hf_model)
        _require_finite(latent_vec, "latent realignment")
        latent_embed = latent_vec.unsqueeze(1)
        past_len = get_past_kv_sequence_length(past)
        latent_mask = torch.ones(
            (latent_embed.shape[0], past_len + 1),
            dtype=torch.long,
            device=latent_embed.device,
        )
        outputs = hf_model(
            inputs_embeds=latent_embed,
            attention_mask=latent_mask,
            past_key_values=past,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        curr_output_embedding.append(latent_embed.detach())

    return past, torch.cat(curr_output_embedding, dim=1)
