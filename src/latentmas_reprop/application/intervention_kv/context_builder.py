"""Latent context builder for intervention experiments."""

import contextlib
import time
from typing import Any

import torch
from tqdm import tqdm

from ...domain.ports.cache_port import CachePort
from ...domain.services.kv_cache import (
    estimate_past_kv_bytes,
    get_past_kv_sequence_length,
    move_past_kv,
)
from ...domain.services.latent_mas import LatentMASMethod
from ...infrastructure.paths.resolver import get_path_resolver
from ...infrastructure.storage import (
    is_condition_complete,
    load_json,
    load_tensor,
    save_condition_result,
)


def build_intervention_contexts(
    method: LatentMASMethod,
    dataset_iter: list[dict],
    task: str,
    seed: int,
    save_raw_cache: bool = False,
    cache_port: CachePort | None = None,
    sample_keys: list[str] | None = None,
    split: str = "test",
    run_id: str | None = None,
    **kwargs: Any,
) -> tuple[list[Any], list[list[dict]], list[float], list[int], list[int]]:
    """Phase 1: Build latent contexts on CPU for each sample in the dataset."""
    contexts: list[Any] = []
    traces_list: list[list[dict]] = []
    context_build_latencies: list[float] = []
    context_seq_lens: list[int] = []
    cache_bytes_list: list[int] = []

    resolver = get_path_resolver()

    for idx, item in enumerate(tqdm(dataset_iter, desc="Building latent contexts")):
        shared_dir = (
            resolver.get_condition_dir(run_id, idx, "shared") if run_id else None
        )
        if shared_dir and is_condition_complete(shared_dir):
            with contextlib.suppress(Exception):
                past_kv_cpu = load_tensor(shared_dir / "kv.pt", map_location="cpu")
                meta = load_json(shared_dir / "result.json")
                contexts.append(past_kv_cpu)
                traces_list.append(meta.get("traces", []))
                context_build_latencies.append(meta.get("build_latency_sec", 0.0))
                context_seq_lens.append(
                    meta.get("seq_len", get_past_kv_sequence_length(past_kv_cpu))
                )
                cache_bytes_list.append(
                    meta.get("cache_bytes", estimate_past_kv_bytes(past_kv_cpu))
                )
                continue

        t_start = time.time()
        past_kv, agent_traces = method.build_latent_contexts([item])
        t_build = time.time() - t_start

        past_kv_cpu = move_past_kv(past_kv, "cpu")
        seq_len = get_past_kv_sequence_length(past_kv_cpu)
        c_bytes = estimate_past_kv_bytes(past_kv_cpu)
        traces = agent_traces[0] if agent_traces else []

        contexts.append(past_kv_cpu)
        traces_list.append(traces)
        context_build_latencies.append(t_build)
        context_seq_lens.append(seq_len)
        cache_bytes_list.append(c_bytes)

        if shared_dir:
            with contextlib.suppress(Exception):
                save_condition_result(
                    shared_dir,
                    {
                        "seq_len": seq_len,
                        "cache_bytes": c_bytes,
                        "build_latency_sec": t_build,
                        "traces": traces,
                    },
                    tensors={"kv": past_kv_cpu},
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
