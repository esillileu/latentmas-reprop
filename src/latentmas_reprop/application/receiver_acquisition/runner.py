"""Execution runner for receiver acquisition experiments."""

import contextlib
from typing import Any

import torch
from tqdm import tqdm

from ...domain.models import ReceiverAcquisitionRecord
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...infrastructure.models.model_wrapper import ModelWrapper
from .forward import build_sender_cache, forward_receiver_observation
from .sampling import SecretDigitSample, SenderCacheBundle


def _resolve_condition_cache(
    mode: str,
    condition: str,
    target_bundle: SenderCacheBundle,
    source_bundle: SenderCacheBundle | None,
) -> Any:
    """Resolve the cache to use for a given context mode and condition."""
    if condition in {"drop", "drop_position_matched"}:
        return None
    bundle = target_bundle if condition == "own" else source_bundle
    if bundle is None:
        return None
    if mode in {
        "latent_only",
        "latent_only_position_fixed",
        "latent_only_compact_debug",
    }:
        return bundle.latent_only
    if mode == "prompt_only":
        return bundle.prompt_only
    return bundle.full


def run_acquisition_loop(
    model: ModelWrapper,
    args: Any,
    samples: list[SecretDigitSample],
    cross_indices: list[int],
    receiver_ids: torch.Tensor,
    receiver_mask: torch.Tensor,
    mapping: dict[str, Any],
    tracker_port: ExperimentTrackerPort | None,
) -> tuple[list[ReceiverAcquisitionRecord], dict[str, Any], float]:
    """Execute sender caching and receiver evaluation loop across all samples and conditions."""
    records: list[ReceiverAcquisitionRecord] = []
    hidden: dict[str, torch.Tensor] = {}
    bundles = [build_sender_cache(model, args, sample) for sample in samples]
    context_runtime = sum(bundle.build_latency_sec for bundle in bundles)

    for index, target in enumerate(tqdm(samples, desc="Receiver evaluation")):
        cross = samples[cross_indices[index]] if cross_indices else None
        target_bundle = bundles[index]
        source_bundle = bundles[cross_indices[index]] if cross_indices else None
        full_len = target_bundle.full_len

        trace_ctx = (
            tracker_port.start_sample_trace(
                "secret_digit_sample",
                {
                    "sample_index": index,
                    "target_digit": target.digit,
                    "cross_digit": cross.digit if cross else None,
                },
                tags={
                    "task": args.task,
                    "model": args.model_name,
                    "sample_id": target.sample_id,
                    "target_digit": str(target.digit),
                },
                request_preview=f"target={target.digit} cross={cross.digit if cross else 'none'}",
            )
            if tracker_port
            else contextlib.nullcontext(None)
        )

        with trace_ctx as root:
            if tracker_port:
                with tracker_port.start_span(
                    "build_sender_context",
                    "CHAIN",
                    {
                        "sample_id": target.sample_id,
                        "digit": target.digit,
                        "latent_steps": args.latent_steps,
                    },
                ) as span:
                    if span is not None:
                        span.set_outputs(
                            {
                                "cache_present": True,
                                "cache_sequence_length": target_bundle.full_len,
                                "latency_sec": target_bundle.build_latency_sec,
                            }
                        )

            for condition in args.acquisition_conditions:
                if condition == "cross" and cross is None:
                    continue
                source = cross if condition == "cross" else None
                modes = (
                    ["none"]
                    if condition in {"drop", "drop_position_matched"}
                    else args.context_modes
                )
                for mode in modes:
                    cache = _resolve_condition_cache(
                        mode, condition, target_bundle, source_bundle
                    )
                    record, state_hidden = forward_receiver_observation(
                        model=model,
                        args=args,
                        target=target,
                        source=source,
                        mode=mode,
                        condition=condition,
                        cache=cache,
                        receiver_ids=receiver_ids,
                        receiver_mask=receiver_mask,
                        mapping=mapping,
                        original_full_seq_len=full_len,
                        tracker_port=tracker_port,
                    )
                    records.append(record)
                    if record.hidden_state_artifact_key and state_hidden is not None:
                        hidden[record.hidden_state_artifact_key] = torch.stack(
                            [t.cpu() for t in state_hidden]
                        )

            sample_records = [r for r in records if r.sample_index == index]
            if root is not None:
                root.set_outputs(
                    {
                        "cells": {
                            f"{r.context_mode}/{r.condition}": {
                                "predicted_digit": r.predicted_digit,
                                "source_probability": r.source_probability,
                            }
                            for r in sample_records
                        }
                    }
                )
                trace_id = root.trace_id
                if trace_id and tracker_port:
                    tracker_port.log_expectation(
                        trace_id, "expected_target_digit", target.digit
                    )
                    if cross:
                        tracker_port.log_expectation(
                            trace_id, "expected_cross_source_digit", cross.digit
                        )
                    for r in sample_records:
                        if r.content_follow_correct is not None:
                            tracker_port.log_feedback(
                                trace_id,
                                f"{r.context_mode}_{r.condition}_source_follow",
                                r.content_follow_correct,
                            )
                    errors = [r.error for r in sample_records if r.error]
                    tracker_port.log_feedback(
                        trace_id,
                        "execution_success",
                        not errors,
                        rationale="; ".join(errors) if errors else None,
                    )

    return records, hidden, context_runtime
