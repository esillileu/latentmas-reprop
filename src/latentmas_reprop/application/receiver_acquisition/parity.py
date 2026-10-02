"""Exact full20-prefix versus fresh independent k-step rollout parity."""

import torch

from ...domain.services.kv_cache import clone_past_kv, move_past_kv
from .forward import build_sender_cache
from .trajectory import acquisition_args, latent_prefix


def cache_pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(cache)


def caches_equal(left, right):
    left_pairs, right_pairs = cache_pairs(left), cache_pairs(right)
    return len(left_pairs) == len(right_pairs) and all(
        torch.equal(lk, rk) and torch.equal(lv, rv)
        for (lk, lv), (rk, rv) in zip(left_pairs, right_pairs, strict=True)
    )


def verify_prefix_parity(model, samples, receiver):
    """Compare exact KV, logits and receiver positions at k=1,4."""
    if len(samples) != 10 or {s.digit for s in samples} != set(range(10)):
        raise ValueError("Parity requires one canonical representative of each digit")
    checks = []
    ids, mask, _ = receiver
    with torch.inference_mode():
        for sample in samples:
            full = build_sender_cache(
                model, acquisition_args(model.model_name, 20), sample
            )
            for step in (1, 4):
                independent = build_sender_cache(
                    model, acquisition_args(model.model_name, step), sample
                )
                prefix = latent_prefix(full, step)
                position = full.prompt_len + step
                outputs = [
                    model.forward_next_token_batch(
                        ids,
                        mask,
                        past_key_values=move_past_kv(
                            clone_past_kv(cache), model.device
                        ),
                        position_start=start,
                    ).logits
                    for cache, start in (
                        (prefix, position),
                        (independent.latent_only, independent.full_len),
                    )
                ]
                checks.append(
                    {
                        "sample_id": sample.sample_id,
                        "digit": sample.digit,
                        "step": step,
                        "cache_exact": caches_equal(prefix, independent.latent_only),
                        "logits_exact": torch.equal(*outputs),
                        "position_exact": position == independent.full_len,
                        "receiver_position_start": position,
                        "independent_receiver_position_start": independent.full_len,
                    }
                )
            print(f"Parity: digit {sample.digit}", flush=True)
    return {
        "passed": all(
            c["cache_exact"] and c["logits_exact"] and c["position_exact"]
            for c in checks
        ),
        "checks": checks,
        "reference": "fresh independent k-step rollout",
    }
