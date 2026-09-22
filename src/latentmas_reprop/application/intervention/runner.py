"""Execution runner for latent communication intervention experiments."""

import random
from typing import Any

import torch
from tqdm import tqdm

from ...domain.models import InterventionCondition, SampleInterventionRecord
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...domain.services.kv_cache import (
    get_past_kv_dtype,
    get_past_kv_num_layers,
)
from ...domain.services.latent_mas import LatentMASMethod
from ...infrastructure.models.model_wrapper import ModelWrapper
from .artifacts import log_sample_assessments
from .decoder import (
    build_sample_root_context,
    decode_single_condition,
    prepare_condition_context,
)

VALID_CONDITIONS = {
    InterventionCondition.OWN,
    InterventionCondition.CROSS,
    InterventionCondition.DROP,
    InterventionCondition.ZERO,
}


def _finalize_root_span(
    root_span: Any,
    tracker_port: ExperimentTrackerPort | None,
    i: int,
    s_key: str,
    conditions: list[str],
    cond_preds: dict[str, tuple[str | None, bool, str]],
) -> str | None:
    """Set root span outputs and update trace response preview."""
    if root_span is None:
        return None
    root_outputs = {"sample_index": i, "sample_key": s_key}
    for c in conditions:
        if c in cond_preds:
            root_outputs[f"{c}_prediction"] = cond_preds[c][0]
            root_outputs[f"{c}_correct"] = cond_preds[c][1]
    root_span.set_outputs(root_outputs)
    if tracker_port:
        resp = " | ".join(
            f"{c}: {cond_preds[c][0]} ({'PASS' if cond_preds[c][1] else 'FAIL'})"
            for c in conditions
            if c in cond_preds
        )
        tracker_port.update_current_trace(response_preview=resp)
    return getattr(root_span, "trace_id", None)


def run_intervention_decoding(
    model: ModelWrapper,
    method: LatentMASMethod,
    args: Any,
    conditions: list[str],
    cross_indices: list[int],
    dataset_iter: list[dict],
    sample_keys: list[str],
    contexts: list[Any],
    traces_list: list[list[dict]],
    context_build_latencies: list[float],
    context_seq_lens: list[int],
    cache_bytes_list: list[int],
    tracker_port: ExperimentTrackerPort | None,
    git_hash: str,
) -> tuple[
    list[SampleInterventionRecord], dict[str, list[float]], dict[str, list[int]]
]:
    """Phase 2: Decode under all intervention conditions and collect sample records."""
    sample_records: list[SampleInterventionRecord] = []
    condition_latencies: dict[str, list[float]] = {c: [] for c in conditions}
    condition_tokens: dict[str, list[int]] = {c: [] for c in conditions}

    num_layers = get_past_kv_num_layers(contexts[0]) if contexts else 0
    cache_dtype = get_past_kv_dtype(contexts[0]) if contexts else None

    for i, item in enumerate(tqdm(dataset_iter, desc="Intervention decoding")):
        sample_id = f"sample_{i}"
        s_key = sample_keys[i]
        gold = str(item.get("gold", ""))
        question = item.get("question", "")
        cross_j = cross_indices[i] if "cross" in conditions else None
        target_seq_len = context_seq_lens[i]
        cond_preds: dict[str, tuple[str | None, bool, str]] = {}
        sample_results: dict[str, SampleInterventionRecord] = {}

        trace_ctx, own_context_gpu, own_traces = build_sample_root_context(
            tracker_port=tracker_port,
            args=args,
            sample_id=sample_id,
            i=i,
            s_key=s_key,
            question=question,
            gold=gold,
            git_hash=git_hash,
            contexts=contexts,
            traces_list=traces_list,
            model_device=model.device,
            target_seq_len=target_seq_len,
            cache_bytes=cache_bytes_list[i],
            num_layers=num_layers,
            cache_dtype=cache_dtype,
            context_build_latency=context_build_latencies[i],
        )

        with trace_ctx as root_span:
            for cond in conditions:
                if cond not in VALID_CONDITIONS:
                    continue
                sample_seed = args.seed + i
                random.seed(sample_seed)
                torch.manual_seed(sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sample_seed)

                (
                    chosen_ctx,
                    chosen_tr,
                    cache_present,
                    src_cache_len,
                ) = prepare_condition_context(
                    cond,
                    i,
                    cross_j,
                    contexts,
                    own_context_gpu,
                    own_traces,
                    model.device,
                    target_seq_len,
                    context_seq_lens,
                )
                rec, latency, gen_tok, c_pred = decode_single_condition(
                    method=method,
                    model=model,
                    item=item,
                    cond=cond,
                    chosen_ctx=chosen_ctx,
                    chosen_tr=chosen_tr,
                    cache_present=cache_present,
                    target_seq_len=target_seq_len,
                    src_cache_len=src_cache_len,
                    cache_delta=src_cache_len - target_seq_len,
                    num_layers=num_layers,
                    cache_dtype=cache_dtype,
                    sample_id=sample_id,
                    sample_index=i,
                    s_key=s_key,
                    cross_j=cross_j,
                    cross_source_key=sample_keys[cross_j]
                    if cross_j is not None
                    else None,
                    sample_seed=sample_seed,
                    args=args,
                    tracker_port=tracker_port,
                )
                sample_records.append(rec)
                sample_results[cond] = rec
                condition_latencies[cond].append(latency)
                condition_tokens[cond].append(gen_tok)
                cond_preds[cond] = c_pred

            del own_context_gpu
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            trace_id = _finalize_root_span(
                root_span, tracker_port, i, s_key, conditions, cond_preds
            )

        log_sample_assessments(
            tracker_port,
            trace_id,
            item,
            gold,
            conditions,
            cond_preds,
            sample_results,
        )

    return sample_records, condition_latencies, condition_tokens
