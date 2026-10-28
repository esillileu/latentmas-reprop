"""Condition decoding and sample record creation for intervention experiments."""

import contextlib
import time
from typing import Any

import torch

from ...domain.models import InterventionCondition, SampleInterventionRecord
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...domain.services.kv_cache import clone_past_kv, create_zero_past_kv, move_past_kv
from ...domain.services.latent_mas import LatentMASMethod
from ...infrastructure.models.model_wrapper import ModelWrapper


def prepare_condition_context(
    cond: str,
    i: int,
    cross_j: int | None,
    contexts: list[Any],
    own_context_gpu: Any,
    own_traces: list[list[dict]],
    device: Any,
    target_seq_len: int,
    context_seq_lens: list[int],
) -> tuple[Any, list[list[dict]] | None, bool, int]:
    """Resolve the KV-cache, traces, presence, and sequence length for a given condition."""
    if cond == InterventionCondition.OWN:
        return own_context_gpu, own_traces, True, target_seq_len
    if cond == InterventionCondition.CROSS:
        if cross_j is None:
            return None, None, False, 0
        chosen = move_past_kv(clone_past_kv(contexts[cross_j]), device)
        return chosen, None, True, context_seq_lens[cross_j]
    if cond == InterventionCondition.DROP:
        return None, None, False, 0
    if cond == InterventionCondition.ZERO:
        chosen = move_past_kv(create_zero_past_kv(contexts[i]), device)
        return chosen, None, True, target_seq_len
    return None, None, False, 0


def build_sample_root_context(
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    sample_id: str,
    i: int,
    s_key: str,
    question: str,
    gold: str,
    git_hash: str,
    contexts: list[Any],
    traces_list: list[list[dict]],
    model_device: Any,
    target_seq_len: int,
    cache_bytes: int,
    num_layers: int,
    cache_dtype: Any,
    context_build_latency: float,
) -> tuple[Any, Any, list[list[dict]]]:
    """Start root trace, clone own context to GPU, and log child build span."""
    trace_ctx = (
        tracker_port.start_sample_trace(
            name="latentmas_sample",
            inputs={
                "sample_index": i,
                "sample_key": s_key,
                "question": question,
                "gold": gold,
            },
            tags={
                "task": args.task,
                "model": args.model_name,
                "method": args.method,
                "sample_id": sample_id,
                "sample_key": s_key,
                "pilot": "true",
                "latent_steps": str(args.latent_steps),
                "seed": str(args.seed),
                "git_commit": git_hash,
            },
            request_preview=f"[{sample_id}] {question[:90]}",
        )
        if tracker_port
        else contextlib.nullcontext(None)
    )

    own_ctx_cloned = clone_past_kv(contexts[i])
    own_context_gpu = move_past_kv(own_ctx_cloned, model_device)
    own_traces = [traces_list[i]]

    if tracker_port:
        with tracker_port.start_span(
            name="build_own_context",
            span_type="CHAIN",
            inputs={
                "sample_id": sample_id,
                "sample_key": s_key,
                "question": question,
                "model": args.model_name,
                "latent_steps": args.latent_steps,
            },
        ) as b_span:
            if b_span is not None:
                b_span.set_outputs(
                    {
                        "cache_present": True,
                        "sample_key": s_key,
                        "cache_sequence_length": target_seq_len,
                        "cache_bytes": cache_bytes,
                        "num_layers": num_layers,
                        "cache_dtype": cache_dtype,
                        "build_latency_sec": round(context_build_latency, 4),
                    }
                )

    return trace_ctx, own_context_gpu, own_traces


def decode_single_condition(
    *,
    method: LatentMASMethod,
    model: ModelWrapper,
    item: dict,
    cond: str,
    chosen_ctx: Any,
    chosen_tr: list[list[dict]] | None,
    cache_present: bool,
    target_seq_len: int,
    src_cache_len: int,
    cache_delta: int,
    num_layers: int,
    cache_dtype: Any,
    sample_id: str,
    sample_index: int,
    s_key: str,
    cross_j: int | None,
    cross_source_key: str | None,
    sample_seed: int,
    args: Any,
    tracker_port: ExperimentTrackerPort | None,
) -> tuple[SampleInterventionRecord, float, int, tuple[str | None, bool, str]]:
    """Decode a single condition, log LLM span metrics, and produce a SampleInterventionRecord."""
    is_cross = cond == "cross" and cross_j is not None
    src_sample_id = (
        sample_id if cond == "own" else f"sample_{cross_j}" if is_cross else None
    )
    src_s_key = s_key if cond == "own" else cross_source_key if is_cross else None

    span_ctx = (
        tracker_port.start_span(
            name=f"decode_{cond}",
            span_type="LLM",
            inputs={
                "condition": cond,
                "sample_id": sample_id,
                "sample_key": s_key,
                "model": args.model_name,
                "latent_steps": args.latent_steps,
                "cache_present": cache_present,
                "target_cache_seq_len": target_seq_len,
                "source_cache_seq_len": src_cache_len,
                "cache_seq_len_delta": cache_delta,
            },
        )
        if tracker_port
        else contextlib.nullcontext(None)
    )

    with span_ctx as c_span:
        t0 = time.time()
        decode_res = method.decode_with_context(
            [item], past_kv=chosen_ctx, initial_traces=chosen_tr
        )
        latency = time.time() - t0

        res = decode_res[0]
        pred, raw_pred = res.get("prediction"), res.get("raw_prediction", "")
        is_correct = bool(res.get("correct", False))
        err_msg = res.get("error_msg") or None
        cond_pred = (pred, is_correct, raw_pred)

        prompt_tokens = len(res.get("agents", [{}])[-1].get("input_ids", []))
        gen_tokens = (
            len(model.tokenizer.encode(raw_pred, add_special_tokens=False))
            if hasattr(model, "tokenizer")
            else len(raw_pred.split())
        )

        if c_span is not None:
            c_span.set_token_usage(
                prompt_tokens=prompt_tokens, completion_tokens=gen_tokens
            )
            c_span.set_outputs(
                {
                    "condition": cond,
                    "sample_key": s_key,
                    "prediction": pred,
                    "correct": is_correct,
                    "generated_tokens": gen_tokens,
                    "prompt_tokens": prompt_tokens,
                    "latency_sec": round(latency, 4),
                }
            )
            if err_msg:
                c_span.set_status("ERROR", description=err_msg)

        record = SampleInterventionRecord(
            sample_id=sample_id,
            source_sample_id=src_sample_id,
            condition=cond,
            question=item.get("question", ""),
            gold=str(item.get("gold", "")),
            prediction=pred,
            raw_prediction=raw_pred,
            correct=is_correct,
            model=args.model_name,
            task=args.task,
            latent_steps=args.latent_steps,
            seed=sample_seed,
            sample_index=sample_index,
            sample_key=s_key,
            source_sample_index=sample_index
            if cond == "own"
            else cross_j
            if cond == "cross"
            else None,
            source_sample_key=src_s_key,
            target_cache_seq_len=target_seq_len,
            source_cache_seq_len=src_cache_len,
            cache_seq_len_delta=cache_delta,
            cache_present=cache_present,
            num_layers=num_layers,
            cache_dtype=cache_dtype,
            latency=round(latency, 4),
            generated_tokens=gen_tokens,
            error=err_msg,
        )

    if cond != InterventionCondition.OWN:
        del chosen_ctx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return record, latency, gen_tokens, cond_pred
