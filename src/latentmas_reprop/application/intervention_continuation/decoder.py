"""Decoder utilities for single intervention autoregressive token generation."""

from typing import Any

import torch
from tqdm import tqdm
from transformers import TopKLogitsWarper, TopPLogitsWarper

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    move_past_kv,
)
from .conditions import COMPARISON_CONDITIONS


def get_lm_head(model_module: torch.nn.Module) -> Any:
    """Retrieve output LM head or projection layer from model."""
    if hasattr(model_module, "lm_head"):
        return model_module.lm_head
    if hasattr(model_module, "get_output_embeddings"):
        return model_module.get_output_embeddings()
    raise AttributeError("Cannot find lm_head or output embeddings on model.")


def eos_token_ids(model: Any, tokenizer: Any) -> list[int]:
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos = tokenizer.eos_token_id if eos is None else eos
    return list(eos) if isinstance(eos, (list, tuple)) else [eos]


def generation_settings(model: Any, args: Any) -> dict[str, Any]:
    """Retain single-intervention decoding; comparisons use explicit sampling filters."""
    conditions = getattr(args, "intervention_conditions", None) or []
    if isinstance(conditions, str):
        conditions = conditions.split(",")
    comparison = bool(
        {condition.strip().lower() for condition in conditions}
        & set(COMPARISON_CONDITIONS)
    )
    return {
        "temperature": float(getattr(args, "temperature", 0.0)),
        "top_p": float(getattr(args, "top_p", 1.0)) if comparison else 1.0,
        "top_k": int(getattr(args, "top_k", 0)) if comparison else 0,
        "eos_ids": eos_token_ids(model.model, model.tokenizer)
        if comparison
        else [model.tokenizer.eos_token_id],
    }


@torch.inference_mode()
def decode_tokens_with_hiddens(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating",
    top_k: int = 0,
    eos_ids: list[int] | None = None,
    termination_ids: list[int] | None = None,
    progress_position: int | None = None,
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    """Autoregressive explicit decoding recording the hidden state predicting each token."""
    lm_head = get_lm_head(model)
    inner_model = getattr(model, "model", model)
    ids: list[int] = []
    gpu_hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden
    kv = past_kv
    if max_new_tokens <= 0 or temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("Invalid continuation generation budget or sampling settings")
    stop_ids = set(eos_token_ids(model, tokenizer) if eos_ids is None else eos_ids)
    filters = []
    if top_k:
        filters.append(TopKLogitsWarper(top_k))
    if top_p < 1:
        filters.append(TopPLogitsWarper(top_p))

    progress = tqdm(
        total=max_new_tokens,
        desc=desc,
        unit="token",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        disable=not report_progress,
        position=progress_position,
    )

    reason = "length"
    exhausted = True
    try:
        for _ in range(max_new_tokens):
            gpu_hiddens.append(curr_hidden.detach())
            logits = lm_head(curr_hidden)
            if logits.dim() == 2:
                logits = logits[-1]
            if temperature > 0.0:
                scores = (logits / temperature).unsqueeze(0)
                for processor in filters:
                    scores = processor(None, scores)
                probs = torch.softmax(scores[0], dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_id = int(torch.argmax(logits, dim=-1).item())

            if next_id in stop_ids:
                if termination_ids is not None:
                    termination_ids.append(next_id)
                gpu_hiddens.pop()
                reason = "stop"
                exhausted = False
                break

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                break

            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            outputs = inner_model(
                input_ids=next_tensor,
                attention_mask=None,
                past_key_values=kv,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            kv = outputs.past_key_values
            if (
                hasattr(outputs, "last_hidden_state")
                and outputs.last_hidden_state is not None
            ):
                curr_hidden = outputs.last_hidden_state[0, -1, :]
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states:
                curr_hidden = outputs.hidden_states[-1][0, -1, :]
            else:
                raise AttributeError(
                    "Model outputs do not contain last_hidden_state or hidden_states."
                )
    finally:
        progress.close()

    cpu_hiddens = [h.cpu() for h in gpu_hiddens]
    return ids, cpu_hiddens, reason, exhausted


def run_explicit_generation(
    model: Any,
    kv: Any,
    start_hidden: torch.Tensor,
    args: Any,
    desc: str = "Generating",
) -> tuple[list[int], list[torch.Tensor], str, bool, str]:
    """Run explicit generation loop and decode tokens to string."""
    max_new = int(getattr(args, "max_new_tokens", 256))
    report_progress = not bool(getattr(args, "no_progress", False))
    ids, hiddens, reason, exhausted = decode_tokens_with_hiddens(
        model.model,
        model.device,
        kv,
        start_hidden.to(model.device),
        model.tokenizer,
        max_new,
        **generation_settings(model, args),
        report_progress=report_progress,
        desc=desc,
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
