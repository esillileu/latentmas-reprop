"""Text generation and forward next-token operations for HuggingFace models."""

from time import perf_counter
from typing import Any

import torch
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from ...domain.ports.model_port import NextTokenState
from ...domain.services.kv_cache import get_past_kv_sequence_length


class GenerationProgress(StoppingCriteria):
    """Report token budget usage without changing the generation stop decision."""

    def __init__(self, prompt_length: int, budget: int):
        self.prompt_length = prompt_length
        self.budget = budget
        self.started = self.last_report = perf_counter()
        print(f"[generation] started: 0/{budget} tokens", flush=True)

    def __call__(self, input_ids, scores, **kwargs):
        now = perf_counter()
        count = input_ids.shape[-1] - self.prompt_length
        if now - self.last_report >= 30 or count >= self.budget:
            elapsed = now - self.started
            print(
                f"[generation] {count}/{self.budget} tokens "
                f"({count / self.budget:.1%} budget), "
                f"{elapsed:.0f}s, {count / max(elapsed, 1e-9):.1f} tokens/s",
                flush=True,
            )
            self.last_report = now
        return False


def count_new_token_ids(generated_ids: torch.Tensor, pad_token_id: int | None) -> int:
    """Count ids in a generate() suffix, excluding pad ids."""
    if pad_token_id is not None:
        generated_ids = generated_ids[generated_ids != pad_token_id]
    return int(generated_ids.shape[0])


def generate_text_batch(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
    past_key_values: Any = None,
    report_progress: bool = False,
) -> tuple[list[str], Any, list[int]]:
    """Generate text completions using HuggingFace model.

    The third value is the number of newly generated token ids per row.
    """
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)

    if past_key_values is not None:
        past_len = get_past_kv_sequence_length(past_key_values)
        if past_len > 0:
            past_mask = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

    sampling = {"temperature": temperature, "top_p": top_p} if temperature > 0 else {}
    progress = (
        {
            "stopping_criteria": StoppingCriteriaList(
                [GenerationProgress(input_ids.shape[-1], max_new_tokens)]
            )
        }
        if report_progress
        else {}
    )
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        **sampling,
        **progress,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
        past_key_values=past_key_values,
    )
    sequences = outputs.sequences
    generations: list[str] = []
    token_counts: list[int] = []
    prompt_seq_len = input_ids.shape[1]
    for idx in range(input_ids.shape[0]):
        generated_ids = sequences[idx, prompt_seq_len:]
        token_counts.append(count_new_token_ids(generated_ids, tokenizer.pad_token_id))
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        generations.append(text)
    if report_progress:
        print(f"[generation] finished: {token_counts} tokens", flush=True)
    return generations, outputs.past_key_values, token_counts


def forward_next_token_batch(
    model: torch.nn.Module,
    device: torch.device,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    past_key_values: Any = None,
    output_hidden_states: bool = False,
    position_start: int | None = None,
) -> NextTokenState:
    """Run a single forward pass returning next-token logits and optional hidden state."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    if past_key_values is not None:
        past_len = get_past_kv_sequence_length(past_key_values)
        if past_len:
            prefix = torch.ones(
                (attention_mask.shape[0], past_len),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((prefix, attention_mask), dim=-1)
    position_ids = None
    if position_start is not None:
        position_ids = (
            torch.arange(
                position_start,
                position_start + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
        )
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
        use_cache=False,
        output_hidden_states=output_hidden_states,
        return_dict=True,
    )
    hidden = None
    if output_hidden_states and outputs.hidden_states is not None:
        hidden = tuple(state[:, -1, :].detach() for state in outputs.hidden_states)
    return NextTokenState(
        logits=outputs.logits[:, -1, :].detach(), hidden_states=hidden
    )
