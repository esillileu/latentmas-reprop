"""Latent context builder for intervention experiments."""

import contextlib
import time
from typing import Any

import torch
from tqdm import tqdm

from ...domain.ports.cache_port import CacheLayer, CachePort
from ...domain.services.kv_cache import (
    estimate_past_kv_bytes,
    get_past_kv_sequence_length,
    move_past_kv,
)
from ...domain.services.latent_mas import LatentMASMethod


def build_intervention_contexts(
    method: LatentMASMethod,
    dataset_iter: list[dict],
    task: str,
    seed: int,
    save_raw_cache: bool,
    cache_port: CachePort,
) -> tuple[list[Any], list[list[dict]], list[float], list[int], list[int]]:
    """Phase 1: Build latent contexts on CPU for each sample in the dataset."""
    contexts: list[Any] = []
    traces_list: list[list[dict]] = []
    context_build_latencies: list[float] = []
    context_seq_lens: list[int] = []
    cache_bytes_list: list[int] = []

    for idx, item in enumerate(tqdm(dataset_iter, desc="Building latent contexts")):
        t_start = time.time()
        past_kv, agent_traces = method.build_latent_contexts([item])
        t_build = time.time() - t_start

        past_kv_cpu = move_past_kv(past_kv, "cpu")
        contexts.append(past_kv_cpu)
        traces_list.append(agent_traces[0] if agent_traces else [])
        context_build_latencies.append(t_build)
        context_seq_lens.append(get_past_kv_sequence_length(past_kv_cpu))
        cache_bytes_list.append(estimate_past_kv_bytes(past_kv_cpu))

        if save_raw_cache:
            cache_name = f"latent_{task}_{idx}_s{seed}.pt"
            with contextlib.suppress(Exception):
                cache_port.save_torch(
                    CacheLayer.LATENT_INTERVENTIONS, cache_name, past_kv_cpu
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        contexts,
        traces_list,
        context_build_latencies,
        context_seq_lens,
        cache_bytes_list,
    )
