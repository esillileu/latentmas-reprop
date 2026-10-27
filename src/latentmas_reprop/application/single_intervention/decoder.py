"""Decoder utilities for single intervention autoregressive token generation."""

from typing import Any

import torch

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    move_past_kv,
)


def get_lm_head(model_module: torch.nn.Module) -> Any:
    """Retrieve output LM head or projection layer from model."""
    if hasattr(model_module, "lm_head"):
        return model_module.lm_head
    if hasattr(model_module, "get_output_embeddings"):
        return model_module.get_output_embeddings()
    raise AttributeError("Cannot find lm_head or output embeddings on model.")


@torch.no_grad()
def decode_tokens_with_hiddens(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    """Autoregressive explicit decoding recording the hidden state predicting each token."""
    lm_head = get_lm_head(model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden
    kv = past_kv

    for _ in range(max(1, max_new_tokens)):
        hiddens.append(curr_hidden.detach().cpu())
        logits = lm_head(curr_hidden)
        if logits.dim() == 2:
            logits = logits[-1]
        if temperature > 0.0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(logits, dim=-1).item())

        if next_id == tokenizer.eos_token_id:
            return ids, hiddens[:-1], "stop", False

        ids.append(next_id)
        if len(ids) >= max_new_tokens:
            return ids, hiddens, "length", True

        next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
        past_len = get_past_kv_sequence_length(kv)
        mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
        outputs = model(
            input_ids=next_tensor,
            attention_mask=mask,
            past_key_values=kv,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        kv = outputs.past_key_values
        curr_hidden = outputs.hidden_states[-1][0, -1, :]

    return ids, hiddens, "length", True


def run_explicit_generation(
    model: Any,
    kv: Any,
    start_hidden: torch.Tensor,
    args: Any,
) -> tuple[list[int], list[torch.Tensor], str, bool, str]:
    """Run explicit generation loop and decode tokens to string."""
    max_new = int(getattr(args, "max_new_tokens", 256))
    temp = float(getattr(args, "temperature", 0.0))
    top_p = float(getattr(args, "top_p", 1.0))
    ids, hiddens, reason, exhausted = decode_tokens_with_hiddens(
        model.model,
        model.device,
        kv,
        start_hidden.to(model.device),
        model.tokenizer,
        max_new,
        temperature=temp,
        top_p=top_p,
    )
    text = model.tokenizer.decode(ids, skip_special_tokens=True).strip()
    return ids, hiddens, reason, exhausted, text


@torch.no_grad()
def forward_intervention_prefix(
    model: Any,
    kv_snapshot: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[Any, torch.Tensor]:
    """Forward intervention token sequence (control or prompt) extending common KV."""
    kv = move_past_kv(clone_past_kv(kv_snapshot), model.device)
    past_len = get_past_kv_sequence_length(kv)
    mask = torch.cat(
        [
            torch.ones((1, past_len), dtype=torch.long, device=model.device),
            attention_mask.to(model.device),
        ],
        dim=-1,
    )
    out = model.model(
        input_ids=input_ids.to(model.device),
        attention_mask=mask,
        past_key_values=kv,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    return out.past_key_values, out.hidden_states[-1][0].detach().cpu()
