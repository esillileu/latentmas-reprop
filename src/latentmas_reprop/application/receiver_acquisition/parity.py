"""Exact full20-prefix versus independent rollout parity, with historical scoring."""

import torch

from ...domain.services.kv_cache import clone_past_kv, move_past_kv
from .forward import build_sender_cache
from .trajectory import acquisition_args, latent_prefix, observe


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


def verify_prefix_parity(model, samples, receiver, historical):
    """Check every digit at k=1,4 before allowing trajectory collection."""
    if len(samples) != 10 or {s.digit for s in samples} != set(range(10)):
        raise ValueError("Parity requires one canonical representative of each digit")
    checks = []
    ids, mask, _ = receiver
    with torch.inference_mode():
        for sample in samples:
            full = build_sender_cache(
                model, acquisition_args(model.model_name, 20), sample
            )
            drop = observe(model, sample, 20, None, full.full_len, receiver)
            for step in (1, 4):
                independent = build_sender_cache(
                    model, acquisition_args(model.model_name, step), sample
                )
                prefix = latent_prefix(full, step)
                same_cache = caches_equal(prefix, independent.latent_only)
                position = full.prompt_len + step
                outputs = [
                    model.forward_next_token_batch(
                        ids,
                        mask,
                        past_key_values=move_past_kv(
                            clone_past_kv(cache), model.device
                        ),
                        position_start=position,
                    ).logits
                    for cache in (prefix, independent.latent_only)
                ]
                current = observe(model, sample, step, prefix, position, receiver)
                previous = historical[step]
                own = previous[sample.sample_id, "own"]
                old_drop = previous[sample.sample_id, "drop"]
                same_identity = all(
                    r["sample_key"] == sample.sample_key
                    and r["target_digit"] == sample.digit
                    for r in (own, old_drop)
                )
                same_history = same_identity and (
                    current["candidate_log_probabilities"]
                    == own["candidate_log_probabilities"]
                    and drop["candidate_log_probabilities"]
                    == old_drop["candidate_log_probabilities"]
                    and current["predicted_digit"] == own["predicted_digit"]
                    and drop["predicted_digit"] == old_drop["predicted_digit"]
                    and current["receiver_position_start"]
                    == own["receiver_position_start"]
                    and current["original_full_seq_len"] == own["original_full_seq_len"]
                )
                checks.append(
                    {
                        "sample_id": sample.sample_id,
                        "digit": sample.digit,
                        "step": step,
                        "cache_exact": same_cache,
                        "logits_exact": torch.equal(*outputs),
                        "historical_output_exact": same_history,
                        "current_candidate_log_probabilities": current[
                            "candidate_log_probabilities"
                        ],
                        "historical_candidate_log_probabilities": own[
                            "candidate_log_probabilities"
                        ],
                        "historical_argmax_exact": current["predicted_digit"]
                        == own["predicted_digit"]
                        and drop["predicted_digit"] == old_drop["predicted_digit"],
                        "historical_max_log_probability_delta": max(
                            abs(
                                now["candidate_log_probabilities"][d]
                                - old["candidate_log_probabilities"][d]
                            )
                            for now, old in ((current, own), (drop, old_drop))
                            for d in own["candidate_log_probabilities"]
                        ),
                    }
                )
            print(f"Parity: digit {sample.digit}", flush=True)
    return {
        "passed": all(
            c["cache_exact"] and c["logits_exact"] and c["historical_output_exact"]
            for c in checks
        ),
        "checks": checks,
        "historical_cache_available": False,
        "historical_cache_note": "Raw KV was not saved; cache equality uses a fresh independent k-step rollout.",
    }
