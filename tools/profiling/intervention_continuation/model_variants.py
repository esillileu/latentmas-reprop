from typing import Any

import torch
from tqdm import tqdm

from latentmas_reprop.application.intervention_continuation.decoder import get_lm_head
from latentmas_reprop.domain.services.kv_cache import get_past_kv_sequence_length

from .sampling import _sample_or_greedy


@torch.no_grad()
def decode_v1_model_inner(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v1: model.model)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    inner_model = getattr(model, "model", model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden
    kv = past_kv

    progress = tqdm(
        total=max_new_tokens,
        desc=desc,
        unit="token",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        disable=not report_progress,
    )

    try:
        for _ in range(max(1, max_new_tokens)):
            hiddens.append(curr_hidden.detach().cpu())
            logits = lm_head(curr_hidden)
            next_id = _sample_or_greedy(logits, temperature, top_p)

            if next_id == tokenizer.eos_token_id:
                return ids, hiddens[:-1], "stop", False

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                return ids, hiddens, "length", True

            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            past_len = get_past_kv_sequence_length(kv)
            mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            outputs = inner_model(
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
    finally:
        progress.close()


def decode_v4_inference_mode(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v4: inference_mode)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    with torch.inference_mode():
        lm_head = get_lm_head(model)
        ids: list[int] = []
        hiddens: list[torch.Tensor] = []
        curr_hidden = initial_hidden
        kv = past_kv

        progress = tqdm(
            total=max_new_tokens,
            desc=desc,
            unit="token",
            dynamic_ncols=True,
            mininterval=0.5,
            leave=False,
            disable=not report_progress,
        )

        try:
            for _ in range(max(1, max_new_tokens)):
                hiddens.append(curr_hidden.detach().cpu())
                logits = lm_head(curr_hidden)
                next_id = _sample_or_greedy(logits, temperature, top_p)

                if next_id == tokenizer.eos_token_id:
                    return ids, hiddens[:-1], "stop", False

                ids.append(next_id)
                progress.update(1)
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
        finally:
            progress.close()


@torch.no_grad()
def decode_v5_mask_optimization(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v5: mask opt)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden
    kv = past_kv

    # For causal auto-regressive generation without left padding, attention_mask=None is valid
    # or pre-allocated. In transformers causal LM, passing attention_mask=None when input_ids is 1 token
    # is supported and avoids reallocating a mask of size (1, past_len + 1) every step.
    progress = tqdm(
        total=max_new_tokens,
        desc=desc,
        unit="token",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        disable=not report_progress,
    )

    try:
        for _ in range(max(1, max_new_tokens)):
            hiddens.append(curr_hidden.detach().cpu())
            logits = lm_head(curr_hidden)
            next_id = _sample_or_greedy(logits, temperature, top_p)

            if next_id == tokenizer.eos_token_id:
                return ids, hiddens[:-1], "stop", False

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                return ids, hiddens, "length", True

            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            # Pass attention_mask=None for single-token causal decoding
            outputs = model(
                input_ids=next_tensor,
                attention_mask=None,
                past_key_values=kv,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            kv = outputs.past_key_values
            curr_hidden = outputs.hidden_states[-1][0, -1, :]

        return ids, hiddens, "length", True
    finally:
        progress.close()


@torch.no_grad()
def decode_v6_last_hidden_direct(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v6: last hidden direct)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    inner_model = getattr(model, "model", model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden
    kv = past_kv

    progress = tqdm(
        total=max_new_tokens,
        desc=desc,
        unit="token",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=False,
        disable=not report_progress,
    )

    try:
        for _ in range(max(1, max_new_tokens)):
            hiddens.append(curr_hidden.detach().cpu())
            logits = lm_head(curr_hidden)
            next_id = _sample_or_greedy(logits, temperature, top_p)

            if next_id == tokenizer.eos_token_id:
                return ids, hiddens[:-1], "stop", False

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                return ids, hiddens, "length", True

            next_tensor = torch.tensor([[next_id]], device=device, dtype=torch.long)
            past_len = get_past_kv_sequence_length(kv)
            mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            # Query inner_model with output_hidden_states=False, grabbing last_hidden_state
            outputs = inner_model(
                input_ids=next_tensor,
                attention_mask=mask,
                past_key_values=kv,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            kv = outputs.past_key_values
            curr_hidden = outputs.last_hidden_state[0, -1, :]

        return ids, hiddens, "length", True
    finally:
        progress.close()
