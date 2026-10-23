"""Latent context builder for intervention experiments."""

import time
from typing import Any

import torch
from tqdm import tqdm

from ...domain.models import compute_sample_key
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
    config_id: str | None = None,
    config_hash: str | None = None,
    sample_keys: list[str] | None = None,
    split: str = "test",
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

        # Content-based stable sample identity
        if sample_keys and idx < len(sample_keys):
            s_key = sample_keys[idx]
        else:
            s_key = compute_sample_key(task, split, item.get("question", ""))
        key_prefix = s_key[:16]

        if save_raw_cache:
            import logging
            logger = logging.getLogger(__name__)

            model_slug = getattr(getattr(method, "model", None), "model_name", "model")
            model_slug = str(model_slug).replace("/", "_").replace("\\", "_")
            latent_steps = getattr(method, "latent_steps", 0)

            # Isolated run cache name using config_hash AND content-based sample_key
            # to guarantee reproducibility regardless of dataset traversal order or subset slicing
            cfg_part = f"_cfg{config_hash}" if config_hash else f"_ls{latent_steps}"
            isolated_name = f"latent_{task}_{model_slug}{cfg_part}_key_{key_prefix}_s{seed}.pt"
            legacy_name = f"latent_{task}_{idx}_s{seed}.pt"
            try:
                cache_port.save_torch(
                    CacheLayer.LATENT_INTERVENTIONS, isolated_name, past_kv_cpu
                )
                # Keep legacy file for backward compatibility
                cache_port.save_torch(
                    CacheLayer.LATENT_INTERVENTIONS, legacy_name, past_kv_cpu
                )
            except Exception as exc:
                logger.warning("Failed to save raw latent context %s: %s", isolated_name, exc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (
        contexts,
        traces_list,
        context_build_latencies,
        context_seq_lens,
        cache_bytes_list,
    )
