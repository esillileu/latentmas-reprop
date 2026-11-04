from typing import Any

import torch
from tqdm import tqdm

from latentmas_reprop.application.intervention_continuation.decoder import get_lm_head
from latentmas_reprop.domain.services.kv_cache import get_past_kv_sequence_length

from .sampling import _sample_or_greedy

_COMPILED_MODEL_CACHE: dict[int, Any] = {}


def get_compiled_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    model_id = id(model)
    if model_id not in _COMPILED_MODEL_CACHE:
        inner = getattr(model, "model", model)
        # compile with dynamic=True or default
        _COMPILED_MODEL_CACHE[model_id] = torch.compile(inner, mode="reduce-overhead")
    return _COMPILED_MODEL_CACHE[model_id]


@torch.no_grad()
def decode_v8_torch_compile(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v8: torch.compile)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    inner_model = get_compiled_inner_model(model)
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
                output_hidden_states=False,
                return_dict=True,
            )
            kv = outputs.past_key_values
            curr_hidden = outputs.last_hidden_state[0, -1, :]

        return ids, hiddens, "length", True
    finally:
        progress.close()
