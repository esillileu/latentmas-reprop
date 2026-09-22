"""Chat prompt formatting and tensor preparation utilities."""

from typing import Any

import torch
from transformers import AutoTokenizer


def render_chat(
    tokenizer: AutoTokenizer,
    messages: list[dict],
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render chat messages to string using tokenizer template or fallback."""
    tpl = getattr(tokenizer, "chat_template", None)
    if tpl:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **(chat_template_kwargs or {}),
        )
    segments = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
    if add_generation_prompt:
        segments.append("<|assistant|>")
    return "\n".join(segments)


def prepare_chat_input(
    tokenizer: AutoTokenizer,
    device: torch.device,
    messages: list[dict],
    add_generation_prompt: bool = True,
) -> tuple[str, torch.Tensor, torch.Tensor, list[str]]:
    """Format single conversation messages into input tensors."""
    prompt_text = render_chat(
        tokenizer, messages, add_generation_prompt=add_generation_prompt
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    active_ids = input_ids[0][attention_mask[0].bool()].tolist()
    tokens = tokenizer.convert_ids_to_tokens(active_ids)
    return prompt_text, input_ids, attention_mask, tokens


def prepare_chat_batch(
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_messages: list[list[dict]],
    add_generation_prompt: bool = True,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[list[str]]]:
    """Format batch of conversation messages into padded input tensors."""
    prompts: list[str] = [
        render_chat(
            tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs=chat_template_kwargs,
        )
        for messages in batch_messages
    ]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    tokens_batch: list[list[str]] = [
        tokenizer.convert_ids_to_tokens(ids_row[mask_row.bool()].tolist())
        for ids_row, mask_row in zip(input_ids, attention_mask, strict=False)
    ]
    return prompts, input_ids, attention_mask, tokens_batch
