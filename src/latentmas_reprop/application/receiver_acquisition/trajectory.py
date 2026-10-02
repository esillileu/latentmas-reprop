"""Stepwise latent-only KV acquisition on the canonical balanced receiver set."""

from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import torch

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    retain_past_kv_prefix,
    truncate_past_kv,
)
from .forward import forward_receiver_observation
from .sender import build_sender_cache


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


def observe(
    model, sample, step, cache, full_len, receiver, tracker_port=None, raw_logits=None
):
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
        tracker_port=tracker_port,
        raw_logits=raw_logits,
    )
    return asdict(record)


def collect_trajectory(model, samples, receiver, records_path, tracker_port=None):
    """Build one full20 cache per sample; evaluate own k=1..20 and one drop."""
    from json import dumps

    rows = []
    outputs_dir = records_path.parent / "receiver_outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    with (
        records_path.open("w", encoding="utf-8") as stream,
        records_path.with_name("sender_contexts.jsonl").open(
            "w", encoding="utf-8"
        ) as contexts,
        torch.inference_mode(),
    ):
        for index, sample in enumerate(samples):
            trace = (
                tracker_port.start_sample_trace(
                    "secret_digit_sample",
                    {
                        "sample_index": index,
                        "target_digit": sample.digit,
                        "latent_steps": 20,
                    },
                    tags={
                        "task": "secret_digit",
                        "model": model.model_name,
                        "sample_id": sample.sample_id,
                        "target_digit": str(sample.digit),
                    },
                    request_preview=f"target={sample.digit} latent_steps=1..20",
                )
                if tracker_port
                else nullcontext(None)
            )
            with trace as root:
                sender_span = (
                    tracker_port.start_span(
                        "build_sender_context",
                        "CHAIN",
                        {
                            "sample_id": sample.sample_id,
                            "digit": sample.digit,
                            "latent_steps": 20,
                        },
                    )
                    if tracker_port
                    else nullcontext(None)
                )
                with sender_span as span:
                    bundle = build_sender_cache(
                        model, acquisition_args(model.model_name, 20), sample
                    )
                    contexts.write(
                        dumps(
                            {
                                "sample_id": sample.sample_id,
                                "sample_key": sample.sample_key,
                                "digit": sample.digit,
                                "model": model.model_name,
                                "latent_steps": 20,
                                **bundle.prompt_input,
                                "cache_sequence_length": bundle.full_len,
                                "prompt_len": bundle.prompt_len,
                                "latency_sec": bundle.build_latency_sec,
                            }
                        )
                        + "\n"
                    )
                    contexts.flush()
                    if span is not None:
                        span.set_inputs(
                            {
                                "sample_id": sample.sample_id,
                                "digit": sample.digit,
                                "latent_steps": 20,
                                **bundle.prompt_input,
                            }
                        )
                        span.set_outputs(
                            {
                                "cache_present": True,
                                "cache_sequence_length": bundle.full_len,
                                "prompt_len": bundle.prompt_len,
                                "latency_sec": bundle.build_latency_sec,
                            }
                        )
                if root is not None:
                    root.set_inputs(
                        {
                            "sample_index": index,
                            "sample_id": sample.sample_id,
                            "sample_key": sample.sample_key,
                            "target_digit": sample.digit,
                            "handoff_cuts": list(range(1, 21)),
                            "sender": bundle.prompt_input,
                            "receiver_prompt": receiver[2]["prompt"],
                            "receiver_messages": receiver[2]["messages"],
                        }
                    )
                sample_rows = []
                raw_logits = {}
                failure = None
                try:
                    drop = observe(
                        model,
                        sample,
                        20,
                        None,
                        bundle.full_len,
                        receiver,
                        tracker_port,
                        raw_logits,
                    )
                    stream.write(dumps(drop) + "\n")
                    rows.append(drop)
                    sample_rows.append(drop)
                    if drop["error"]:
                        raise RuntimeError(drop["error"])
                    for step in range(1, 21):
                        cache = latent_prefix(bundle, step)
                        own = observe(
                            model,
                            sample,
                            step,
                            cache,
                            bundle.prompt_len + step,
                            receiver,
                            tracker_port,
                            raw_logits,
                        )
                        stream.write(dumps(own) + "\n")
                        rows.append(own)
                        sample_rows.append(own)
                        if own["error"]:
                            raise RuntimeError(own["error"])
                except Exception as exc:
                    if root is not None:
                        root.set_status("ERROR", str(exc))
                    failure = exc
                finally:
                    torch.save(raw_logits, outputs_dir / f"{sample.sample_id}.pt")
                    if root is not None:
                        root.set_outputs(
                            {
                                "cells": {
                                    (
                                        "none/drop"
                                        if r["condition"] == "drop"
                                        else f"latent_only/own/step_{r['latent_steps']}"
                                    ): r
                                    for r in sample_rows
                                }
                            }
                        )
            if root is not None and root.trace_id and tracker_port:
                tracker_port.log_expectation(
                    root.trace_id, "expected_target_digit", sample.digit
                )
                for r in sample_rows:
                    if r["content_follow_correct"] is not None:
                        name = (
                            "none_drop_source_follow"
                            if r["condition"] == "drop"
                            else f"latent_only_own_step_{r['latent_steps']}_source_follow"
                        )
                        tracker_port.log_feedback(
                            root.trace_id, name, r["content_follow_correct"]
                        )
                tracker_port.log_feedback(
                    root.trace_id, "execution_success", failure is None
                )
            if failure is not None:
                stream.flush()
                raise failure
            stream.flush()
            print(f"{model.model_name}: {index + 1}/{len(samples)} samples", flush=True)
    return rows
