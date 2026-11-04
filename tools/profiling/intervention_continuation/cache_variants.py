from typing import Any

import torch
from tqdm import tqdm
from transformers.cache_utils import StaticCache

from latentmas_reprop.application.intervention_continuation.decoder import get_lm_head
from latentmas_reprop.domain.services.kv_cache import get_past_kv_sequence_length

from .sampling import _sample_or_greedy


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
