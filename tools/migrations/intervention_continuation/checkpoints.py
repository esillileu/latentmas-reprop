"""Validate and split legacy cumulative intervention checkpoints."""

from dataclasses import dataclass
from typing import Any

import torch

from latentmas_reprop.domain.services.kv_cache import (
    _dense_layers,
    concat_past_kv,
    get_past_kv_sequence_length,
    slice_past_kv,
)


@dataclass(frozen=True)
class Checkpoint:
    step: int
    kv: Any
    hidden: torch.Tensor
    metadata: dict[str, Any]


def equal_kv(left: Any, right: Any) -> bool:
    a, _ = _dense_layers(left)
    b, _ = _dense_layers(right)
    return len(a) == len(b) and all(
        x.dtype == y.dtype and torch.equal(x, y)
        for layer_a, layer_b in zip(a, b, strict=True)
        for x, y in zip(layer_a, layer_b, strict=True)
    )


def split_checkpoints(checkpoints: list[Checkpoint]) -> list[Checkpoint]:
    """Validate cumulative checkpoints and return nonoverlapping segments.

    Each output hidden is the full prefix. No checkpoint is modified.
    """
    if not checkpoints:
        raise ValueError("No checkpoints")
    ordered = sorted(checkpoints, key=lambda item: item.step)
    reference = ordered[-1]
    prompt = int(reference.metadata["prompt_length"])
    if prompt <= 0:
        raise ValueError("Invalid prompt length")
    for item in ordered:
        if item.step < 0 or int(item.metadata["latent_steps"]) != item.step:
            raise ValueError("Invalid latent step metadata")
        if item.metadata.get("prompt_length") != prompt or item.metadata.get(
            "sample_index"
        ) != reference.metadata.get("sample_index"):
            raise ValueError("Conflicting checkpoint sample identity")
        length = prompt + item.step
        if (
            get_past_kv_sequence_length(item.kv) != length
            or not torch.is_tensor(item.hidden)
            or item.hidden.shape[0] != length
        ):
            raise ValueError("Invalid checkpoint length")
        if not equal_kv(item.kv, slice_past_kv(reference.kv, 0, length)):
            raise ValueError("Conflicting KV prefix")
        if item.hidden.dtype != reference.hidden.dtype or not torch.equal(
            item.hidden, reference.hidden[:length]
        ):
            raise ValueError("Conflicting hidden prefix")
    boundaries = sorted(
        {0, *range(10, reference.step + 1, 10), *(c.step for c in ordered)}
    )
    result = []
    previous = 0
    for step in boundaries:
        start = 0 if step == 0 else prompt + previous
        end = prompt + step
        meta = dict(reference.metadata, latent_steps=step, start_step=previous)
        result.append(
            Checkpoint(
                step,
                slice_past_kv(reference.kv, start, end),
                reference.hidden[:end].detach().cpu().clone(),
                meta,
            )
        )
        previous = step
    for item in ordered:
        pieces = [part.kv for part in result if part.step <= item.step]
        if not equal_kv(concat_past_kv(pieces), item.kv):
            raise ValueError("Reassembled KV differs from source")
    return result
