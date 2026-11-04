from typing import Any

import torch
from tqdm import tqdm

from latentmas_reprop.application.intervention_continuation.decoder import get_lm_head
from latentmas_reprop.domain.services.kv_cache import get_past_kv_sequence_length

from .sampling import _sample_or_greedy


@torch.no_grad()
def decode_v2_gpu_buffering(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v2: GPU buffer)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    ids: list[int] = []
    gpu_hiddens: list[torch.Tensor] = []
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

    reason = "length"
    exhausted = True
    try:
        for _ in range(max(1, max_new_tokens)):
            gpu_hiddens.append(curr_hidden.detach())
            logits = lm_head(curr_hidden)
            next_id = _sample_or_greedy(logits, temperature, top_p)

            if next_id == tokenizer.eos_token_id:
                gpu_hiddens.pop()  # discard the hidden state of EOS like baseline
                reason = "stop"
                exhausted = False
                break

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                break

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
    finally:
        progress.close()

    cpu_hiddens = [h.cpu() for h in gpu_hiddens]
    return ids, cpu_hiddens, reason, exhausted


@torch.no_grad()
def decode_v3_gpu_token_tensor(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v3: GPU token tensor)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
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
            if logits.dim() == 2:
                logits = logits[-1]
            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                token_tensor = torch.multinomial(probs, num_samples=1)  # shape (1,)
            else:
                token_tensor = torch.argmax(logits, dim=-1, keepdim=True)  # shape (1,)

            next_id = int(token_tensor.item())

            if next_id == tokenizer.eos_token_id:
                return ids, hiddens[:-1], "stop", False

            ids.append(next_id)
            progress.update(1)
            if len(ids) >= max_new_tokens:
                return ids, hiddens, "length", True

            next_tensor = token_tensor.unsqueeze(0)  # shape (1, 1) directly on device
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


def decode_cumulative_best(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (Cumulative Best)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    with torch.inference_mode():
        lm_head = get_lm_head(model)
        inner_model = getattr(model, "model", model)
        ids: list[int] = []
        gpu_hiddens: list[torch.Tensor] = []
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

        reason = "length"
        exhausted = True
        try:
            for _ in range(max(1, max_new_tokens)):
                gpu_hiddens.append(curr_hidden.detach())
                logits = lm_head(curr_hidden)
                if logits.dim() == 2:
                    logits = logits[-1]
                if temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    token_tensor = torch.multinomial(probs, num_samples=1)
                else:
                    token_tensor = torch.argmax(logits, dim=-1, keepdim=True)

                next_id = int(token_tensor.item())

                if next_id == tokenizer.eos_token_id:
                    gpu_hiddens.pop()
                    reason = "stop"
                    exhausted = False
                    break

                ids.append(next_id)
                progress.update(1)
                if len(ids) >= max_new_tokens:
                    break

                next_tensor = token_tensor.unsqueeze(0)
                outputs = inner_model(
                    input_ids=next_tensor,
                    attention_mask=None,
                    past_key_values=kv,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
                kv = outputs.past_key_values
                curr_hidden = outputs.last_hidden_state[0, -1, :]
        finally:
            progress.close()

        cpu_hiddens = [h.cpu() for h in gpu_hiddens]
        return ids, cpu_hiddens, reason, exhausted
