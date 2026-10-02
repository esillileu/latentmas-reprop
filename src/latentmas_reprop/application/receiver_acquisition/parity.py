"""Exact full20-prefix versus independent rollout parity, with historical scoring."""

import csv
import json

import torch

from ...domain.services.kv_cache import clone_past_kv, move_past_kv
from .forward import build_sender_cache
from .trajectory import acquisition_args, latent_prefix, observe


def load_history(acquisition, reference_dir, model):
    with (acquisition / "receiver_conditions.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    history, sources = {}, {}
    for step in (1, 4):
        matches = [
            c
            for c in cells
            if c["model"] == model
            and c["latent_steps"] == str(step)
            and c["condition"] == "drop"
            and c["sample_count"] == "100"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one canonical historical run for {model}, k={step}"
            )
        run_id = matches[0]["run_id"]
        path = reference_dir / run_id / "sample_results.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        selected = [
            r
            for r in records
            if r["condition"] == "drop"
            or (r["condition"] == "own" and r["context_mode"] == "latent_only")
        ]
        if len(selected) != 200 or any(
            r["model"] != model
            or r["latent_steps"] != step
            or r["error"]
            or r["seed"] != 42
            for r in selected
        ):
            raise ValueError(f"Invalid historical receiver run: {run_id}")
        for r in selected:
            if r["cache_present"] and r["cache_dtype"] != "torch.bfloat16":
                raise ValueError("Historical handoff must use BF16")
        history[step] = {(r["sample_id"], r["condition"]): r for r in selected}
        if len(history[step]) != 200:
            raise ValueError("Duplicate historical sample/condition")
        sources[str(step)] = run_id
    return history, sources


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


def verify_prefix_parity(model, samples, receiver, historical=None):
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
                previous = historical[step] if historical is not None else None
                own = previous[sample.sample_id, "own"] if previous else current
                old_drop = previous[sample.sample_id, "drop"] if previous else drop
                same_identity = all(
                    r["sample_key"] == sample.sample_key
                    and r["target_digit"] == sample.digit
                    for r in (own, old_drop)
                )
                same_history = same_identity and (
                    current["predicted_digit"] == own["predicted_digit"]
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
                        "historical_output_exact": same_history if previous else None,
                        "current_candidate_log_probabilities": current[
                            "candidate_log_probabilities"
                        ],
                        "historical_candidate_log_probabilities": own[
                            "candidate_log_probabilities"
                        ]
                        if previous
                        else None,
                        "historical_argmax_exact": (
                            current["predicted_digit"] == own["predicted_digit"]
                            and drop["predicted_digit"] == old_drop["predicted_digit"]
                        )
                        if previous
                        else None,
                        "historical_max_log_probability_delta": max(
                            abs(
                                now["candidate_log_probabilities"][d]
                                - old["candidate_log_probabilities"][d]
                            )
                            for now, old in ((current, own), (drop, old_drop))
                            for d in own["candidate_log_probabilities"]
                        )
                        if previous
                        else None,
                    }
                )
            print(f"Parity: digit {sample.digit}", flush=True)
    return {
        "passed": all(
            c["cache_exact"]
            and c["logits_exact"]
            and c["historical_output_exact"] is not False
            for c in checks
        ),
        "checks": checks,
        "historical_reference_available": historical is not None,
        "historical_cache_available": False,
        "historical_cache_note": "Raw KV was not saved; cache equality uses a fresh independent k-step rollout.",
    }
