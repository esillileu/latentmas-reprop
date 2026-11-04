"""Optimized decoder variants for intervention continuation profiling.

Each variant maintains the same signature and return contract as
decode_tokens_with_hiddens():
    (model, device, past_kv, initial_hidden, tokenizer, max_new_tokens,
     temperature=0.0, top_p=1.0, report_progress=True, desc="Generating")
    -> tuple[list[int], list[torch.Tensor], str, bool]
"""

from collections.abc import Callable
from typing import Any

import torch
from tqdm import tqdm
from transformers.cache_utils import StaticCache

from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
    get_lm_head,
)
from latentmas_reprop.domain.services.kv_cache import (
    get_past_kv_sequence_length,
)


def _sample_or_greedy(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Helper to pick next token id from logits."""
    if logits.dim() == 2:
        logits = logits[-1]
    if temperature > 0.0:
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                torch.softmax(sorted_logits / temperature, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())
    return int(torch.argmax(logits, dim=-1).item())


# ---------------------------------------------------------
# Variant 1: Call model.model() instead of model()
# (Removes redundant LM Head computation inside forward pass)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 2: GPU buffering of hidden states (batch CPU transfer)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 3: Reuse GPU tensor directly for next_id
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 4: torch.inference_mode()
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 5: Attention Mask optimization (None or in-place expansion)
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 6: output_hidden_states=False & last_hidden_state direct access
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 7: StaticCache
# ---------------------------------------------------------
@torch.no_grad()
def decode_v7_static_cache(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v7: StaticCache)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    inner_model = getattr(model, "model", model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden

    past_len = get_past_kv_sequence_length(past_kv)
    max_cache_len = past_len + max_new_tokens + 16

    config = getattr(model, "config", getattr(inner_model, "config", None))
    static_cache = StaticCache(
        config=config,
        max_cache_len=max_cache_len,
    )

    if hasattr(past_kv, "layers"):
        for i, layer in enumerate(past_kv.layers):
            static_cache.update(layer.keys, layer.values, layer_idx=i)
    elif hasattr(past_kv, "key_cache"):
        for i in range(len(past_kv.key_cache)):
            static_cache.update(
                past_kv.key_cache[i], past_kv.value_cache[i], layer_idx=i
            )
    elif isinstance(past_kv, (tuple, list)):
        for i, (k, v) in enumerate(past_kv):
            static_cache.update(k, v, layer_idx=i)

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
            outputs = inner_model(
                input_ids=next_tensor,
                past_key_values=static_cache,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )
            curr_hidden = outputs.last_hidden_state[0, -1, :]

        return ids, hiddens, "length", True
    finally:
        progress.close()


# ---------------------------------------------------------
# Variant 8: torch.compile()
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Variant 9: CUDA Graph
# (Captures static forward pass with StaticCache)
# ---------------------------------------------------------
class CUDAGraphRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        past_kv: Any,
        max_cache_len: int,
        device: torch.device,
    ):
        self.device = device
        self.inner_model = getattr(model, "model", model)
        self.config = getattr(
            model, "config", getattr(self.inner_model, "config", None)
        )

        self.static_input_ids = torch.zeros((1, 1), dtype=torch.long, device=device)
        self.static_cache = StaticCache(
            config=self.config,
            max_cache_len=max_cache_len,
        )
        if hasattr(past_kv, "layers"):
            for i, layer in enumerate(past_kv.layers):
                self.static_cache.update(layer.keys, layer.values, layer_idx=i)
        elif hasattr(past_kv, "key_cache"):
            for i in range(len(past_kv.key_cache)):
                self.static_cache.update(
                    past_kv.key_cache[i], past_kv.value_cache[i], layer_idx=i
                )
        elif isinstance(past_kv, (tuple, list)):
            for i, (k, v) in enumerate(past_kv):
                self.static_cache.update(k, v, layer_idx=i)

        self.graph = torch.cuda.CUDAGraph()
        self.static_output = None
        self._warmup_and_capture()

    def _warmup_and_capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.inner_model(
                    input_ids=self.static_input_ids,
                    past_key_values=self.static_cache,
                    use_cache=True,
                    output_hidden_states=False,
                    return_dict=True,
                )
            torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(self.graph):
            self.static_output = self.inner_model(
                input_ids=self.static_input_ids,
                past_key_values=self.static_cache,
                use_cache=True,
                output_hidden_states=False,
                return_dict=True,
            )

    def forward(self, input_id: int) -> torch.Tensor:
        self.static_input_ids[0, 0] = input_id
        self.graph.replay()
        return self.static_output.last_hidden_state[0, -1, :]


@torch.no_grad()
def decode_v9_cuda_graph(
    model: torch.nn.Module,
    device: torch.device,
    past_kv: Any,
    initial_hidden: torch.Tensor,
    tokenizer: Any,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    report_progress: bool = True,
    desc: str = "Generating (v9: CUDA Graph)",
) -> tuple[list[int], list[torch.Tensor], str, bool]:
    lm_head = get_lm_head(model)
    ids: list[int] = []
    hiddens: list[torch.Tensor] = []
    curr_hidden = initial_hidden

    past_len = get_past_kv_sequence_length(past_kv)
    max_cache_len = past_len + max_new_tokens + 16

    runner = CUDAGraphRunner(
        model=model,
        past_kv=past_kv,
        max_cache_len=max_cache_len,
        device=device,
    )

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

            curr_hidden = runner.forward(next_id).clone()

        return ids, hiddens, "length", True
    finally:
        progress.close()


# ---------------------------------------------------------
# Cumulative Optimized Variant (v1 + v2 + v3 + v4 + v5 + v6)
# Combines non-invasive safe optimizations
# ---------------------------------------------------------
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


VARIANTS: dict[str, Callable] = {
    "baseline": decode_tokens_with_hiddens,
    "v1_model_inner": decode_v1_model_inner,
    "v2_gpu_buffering": decode_v2_gpu_buffering,
    "v3_gpu_token_tensor": decode_v3_gpu_token_tensor,
    "v4_inference_mode": decode_v4_inference_mode,
    "v5_mask_optimization": decode_v5_mask_optimization,
    "v6_last_hidden_direct": decode_v6_last_hidden_direct,
    "v7_static_cache": decode_v7_static_cache,
    "v8_torch_compile": decode_v8_torch_compile,
    "v9_cuda_graph": decode_v9_cuda_graph,
    "cumulative_best": decode_cumulative_best,
}
