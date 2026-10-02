"""Synchronized budget timings and deterministic receiver work proxies."""

from time import perf_counter

import torch
from transformers import StoppingCriteria


class ReceiverTiming(StoppingCriteria):
    """Observe budget checkpoints without changing token generation decisions."""

    def __init__(self, prompt_length, budgets, device):
        self.prompt_length = prompt_length
        self.budgets = set(budgets)
        self.device = torch.device(device)
        self.checkpoints = {}
        self.started = None

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def start(self):
        self.synchronize()
        self.started = perf_counter()

    def elapsed(self):
        self.synchronize()
        return perf_counter() - self.started

    def __call__(self, input_ids, scores, **kwargs):
        count = input_ids.shape[-1] - self.prompt_length
        if count in self.budgets:
            self.checkpoints[count] = self.elapsed()
        return False


def compute_proxy(cache_positions, prompt_tokens, generated_tokens):
    """Count causal attention query/key pairs and processed transformer positions.

    The prompt prefill emits the first generated token; G-1 positions are
    subsequently decoded. Counts are per layer/head, not FLOPs or wall time.
    """
    c, p, d = cache_positions, prompt_tokens, max(0, generated_tokens - 1)
    return {
        "receiver_attention_pairs": p * c
        + p * (p + 1) // 2
        + d * (c + p)
        + d * (d + 1) // 2,
        "receiver_processed_positions": p + d,
        "receiver_cache_positions": c,
        "receiver_prompt_tokens": p,
    }


def prefix_cost(metadata, generated_tokens, budget):
    timings = metadata["receiver_budget_latency_sec"]
    latency = (
        metadata["receiver_latency_sec"]
        if budget == "free" or generated_tokens < budget
        else timings[str(budget)]
    )
    return {
        "receiver_latency_sec": latency,
        **compute_proxy(
            metadata["receiver_cache_positions"],
            metadata["receiver_prompt_tokens"],
            generated_tokens,
        ),
    }
