"""Stepwise latent-only KV acquisition on the canonical balanced receiver set."""

from dataclasses import asdict
from types import SimpleNamespace

import torch

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    retain_past_kv_prefix,
    truncate_past_kv,
)
from .forward import build_sender_cache, forward_receiver_observation
from .sampling import generate_secret_digit_samples


def latent_prefix(bundle, step):
    """Retain full[:prompt+k] before taking its last k positions."""
    if not 1 <= step <= bundle.full_len - bundle.prompt_len:
        raise ValueError(f"Step outside saved sender rollout: {step}")
    prefix = retain_past_kv_prefix(clone_past_kv(bundle.full), bundle.prompt_len + step)
    cache = truncate_past_kv(prefix, step)
    if get_past_kv_sequence_length(cache) != step:
        raise ValueError("Incorrect latent-only prefix length")
    return cache


def acquisition_args(model_name, step):
    return SimpleNamespace(
        model_name=model_name,
        latent_steps=step,
        task="secret_digit",
        seed=42,
        save_hidden_states=False,
    )


def observe(model, sample, step, cache, full_len, receiver):
    ids, mask, mapping = receiver
    record, _ = forward_receiver_observation(
        model=model,
        args=acquisition_args(model.model_name, step),
        target=sample,
        source=sample,
        mode="latent_only" if cache is not None else "none",
        condition="own" if cache is not None else "drop",
        cache=cache,
        receiver_ids=ids,
        receiver_mask=mask,
        mapping=mapping,
        original_full_seq_len=full_len,
        tracker_port=None,
    )
    if record.error:
        raise RuntimeError(record.error)
    return asdict(record)


def collect_trajectory(model, samples, receiver, records_path):
    """Build one full20 cache per sample; evaluate own k=1..20 and one drop."""
    from json import dumps

    rows = []
    with records_path.open("w", encoding="utf-8") as stream, torch.inference_mode():
        for index, sample in enumerate(samples):
            bundle = build_sender_cache(
                model, acquisition_args(model.model_name, 20), sample
            )
            drop = observe(model, sample, 20, None, bundle.full_len, receiver)
            stream.write(dumps(drop) + "\n")
            rows.append(drop)
            for step in range(1, 21):
                cache = latent_prefix(bundle, step)
                own = observe(
                    model, sample, step, cache, bundle.prompt_len + step, receiver
                )
                stream.write(dumps(own) + "\n")
                rows.append(own)
            stream.flush()
            print(f"{model.model_name}: {index + 1}/{len(samples)} samples", flush=True)
    return rows


def summarize_trajectory(records, probe_cells, model, expected_samples=100):
    """Join paired prediction changes to saved full20 post-realignment probes."""
    drops = {r["sample_id"]: r for r in records if r["condition"] == "drop"}
    if len(drops) != expected_samples:
        raise ValueError("Incomplete or duplicate drop sample set")
    if len(records) != expected_samples * 21 or any(r["error"] for r in records):
        raise ValueError("Incomplete receiver trajectory")
    canonical = {s.sample_id: s for s in generate_secret_digit_samples(100, 42)}
    for row in records:
        sample = canonical.get(row["sample_id"])
        if sample is None or (row["sample_key"], row["target_digit"]) != (
            sample.sample_key,
            sample.digit,
        ):
            raise ValueError("Receiver sample is not from the canonical balanced 100")
        if (
            row["model"] != model
            or row["seed"] != 42
            or row["predicted_digit"] not in range(10)
        ):
            raise ValueError("Incorrect model/seed or invalid receiver prediction")
    probes = [
        r
        for r in probe_cells
        if r["model"] == model
        and r["latent_steps"] == "20"
        and r["representation"] == "latent_post_realign"
    ]
    if len(probes) != 20 or {int(r["step"]) for r in probes} != set(range(1, 21)):
        raise ValueError("Expected unique saved post-realignment probe cells 1..20")
    result = []
    for probe in sorted(probes, key=lambda r: int(r["step"])):
        step = int(probe["step"])
        own = [
            r for r in records if r["condition"] == "own" and r["latent_steps"] == step
        ]
        if len(own) != expected_samples or {r["sample_id"] for r in own} != set(drops):
            raise ValueError(f"Unpaired or duplicate receiver samples at step {step}")
        for r in own:
            baseline = drops[r["sample_id"]]
            if (r["sample_key"], r["target_digit"]) != (
                baseline["sample_key"],
                baseline["target_digit"],
            ):
                raise ValueError("Mismatched drop identity")
            if (
                r["context_mode"] != "latent_only"
                or r["source_digit"] != r["target_digit"]
            ):
                raise ValueError("Expected latent-only own handoff")
            if (
                r["cache_sequence_length"] != step
                or r["receiver_position_start"] != r["original_full_seq_len"]
                or r["retained_tail_start_position"] + step
                != r["original_full_seq_len"]
                or r["retained_original_position_end"] != r["original_full_seq_len"]
            ):
                raise ValueError("Incorrect handoff length or receiver position")
        changed = sum(
            r["predicted_digit"] != drops[r["sample_id"]]["predicted_digit"]
            for r in own
        )
        result.append(
            {
                "model": model,
                "step": step,
                "sample_count": expected_samples,
                "probe_run_id": probe["run_id"],
                "probe_accuracy": float(probe["probe_accuracy"]),
                "null_mean_accuracy": float(probe["null_mean_accuracy"]),
                "probe_effect_pp": 100
                * (float(probe["probe_accuracy"]) - float(probe["null_mean_accuracy"])),
                "probe_significant": probe["significance"].lower() == "true",
                "argmax_changed_fraction": changed / expected_samples,
            }
        )
    return result
