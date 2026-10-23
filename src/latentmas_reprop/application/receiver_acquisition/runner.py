"""Execution runner for receiver acquisition experiments."""

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ...domain.models import ReceiverAcquisitionRecord
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...infrastructure.models.model_wrapper import ModelWrapper
from ...infrastructure.paths.resolver import get_path_resolver
from ...infrastructure.storage import (
    is_condition_complete,
    load_json,
    load_tensor,
    save_condition_result,
)
from .forward import forward_receiver_observation
from .sampling import SecretDigitSample, SenderCacheBundle
from .sender import build_sender_cache


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
    incremental_jsonl_path: Any = None,
    run_id: str | None = None,
) -> tuple[list[ReceiverAcquisitionRecord], dict[str, Any], float]:
    """Execute sender caching and receiver evaluation loop across all samples and conditions."""
    records: list[ReceiverAcquisitionRecord] = []
    hidden: dict[str, torch.Tensor] = {}
    resolver = get_path_resolver()

    # Legacy jsonl checkpoint support
    completed_by_index: dict[int, list[ReceiverAcquisitionRecord]] = {}
    if incremental_jsonl_path:
        chk = Path(incremental_jsonl_path)
        if chk.is_file():
            for line in chk.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                idx = d.get("sample_index")
                if idx is not None:
                    fields = ReceiverAcquisitionRecord.__dataclass_fields__.keys()
                    rec = ReceiverAcquisitionRecord(
                        **{f: d[f] for f in fields if f in d}
                    )
                    completed_by_index.setdefault(idx, []).append(rec)

    bundles = [build_sender_cache(model, args, sample) for sample in samples]
    context_runtime = sum(bundle.build_latency_sec for bundle in bundles)

    for index, target in enumerate(tqdm(samples, desc="Receiver evaluation")):
        cross = samples[cross_indices[index]] if cross_indices else None
        target_bundle = bundles[index]
        source_bundle = bundles[cross_indices[index]] if cross_indices else None
        full_len = target_bundle.full_len

        # Check conditions
        pending_conditions = []
        for condition in args.acquisition_conditions:
            if condition == "cross" and cross is None:
                continue
            modes = (
                ["none"]
                if condition in {"drop", "drop_position_matched"}
                else args.context_modes
            )
            for mode in modes:
                cond_name = (
                    f"{mode}_{condition}" if mode != "none" else str(condition)
                )
                if run_id:
                    cond_dir = resolver.get_condition_dir(
                        run_id, target.sample_id, cond_name
                    )
                    if is_condition_complete(cond_dir):
                        try:
                            d = load_json(cond_dir / "result.json")
                            fields = (
                                ReceiverAcquisitionRecord.__dataclass_fields__.keys()
                            )
                            rec = ReceiverAcquisitionRecord(
                                **{f: d[f] for f in fields if f in d}
                            )
                            records.append(rec)
                            if (
                                rec.hidden_state_artifact_key
                                and "hidden.pt" in d.get("files", [])
                            ):
                                hidden[rec.hidden_state_artifact_key] = load_tensor(
                                    cond_dir / "hidden.pt"
                                )
                            continue
                        except Exception:
                            pass
                elif index in completed_by_index:
                    matched = [
                        r
                        for r in completed_by_index[index]
                        if r.condition == condition and r.context_mode == mode
                    ]
                    if matched:
                        records.append(matched[0])
                        continue

                pending_conditions.append((condition, mode, cond_name))

        if not pending_conditions:
            continue

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
                        **target_bundle.prompt_input,
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

            for condition, mode, cond_name in pending_conditions:
                source = cross if condition == "cross" else target
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

                tensors = None
                if record.hidden_state_artifact_key and state_hidden is not None:
                    st_hidden = torch.stack([t.cpu() for t in state_hidden])
                    hidden[record.hidden_state_artifact_key] = st_hidden
                    tensors = {"hidden": st_hidden}

                if run_id:
                    cond_dir = resolver.get_condition_dir(
                        run_id, target.sample_id, cond_name
                    )
                    save_condition_result(
                        cond_dir, record.to_dict(), tensors=tensors
                    )

            sample_records = [r for r in records if r.sample_index == index]
            if incremental_jsonl_path is not None:
                with open(incremental_jsonl_path, "a", encoding="utf-8") as f:
                    for r in sample_records:
                        f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
                    f.flush()
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
