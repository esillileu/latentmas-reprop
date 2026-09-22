"""Canonical helpers for inspecting and manipulating transformer KV caches."""

import copy
from typing import Any

import torch


def get_past_kv_sequence_length(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    try:
        return int(past_kv[0][0].shape[-2])
    except (IndexError, TypeError, AttributeError):
        return 0


def get_past_kv_num_layers(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "layers"):
        return len(past_kv.layers)
    return len(past_kv) if isinstance(past_kv, (list, tuple)) else 0


def _cache_tensors(past_kv: Any):
    if past_kv is None:
        return
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            for attr in ("keys", "values"):
                tensor = getattr(layer, attr, None)
                if torch.is_tensor(tensor):
                    yield tensor
    elif isinstance(past_kv, (list, tuple)):
        for layer in past_kv:
            values = layer if isinstance(layer, (list, tuple)) else (layer,)
            yield from (value for value in values if torch.is_tensor(value))


def get_past_kv_dtype(past_kv: Any) -> str | None:
    return next((str(tensor.dtype) for tensor in _cache_tensors(past_kv)), None)


def estimate_past_kv_bytes(past_kv: Any) -> int:
    return sum(
        tensor.nelement() * tensor.element_size() for tensor in _cache_tensors(past_kv)
    )


def clone_past_kv(past_kv: Any) -> Any:
    """Deep-copy a cache. Call before helpers that mutate DynamicCache objects."""
    return copy.deepcopy(past_kv) if past_kv is not None else None


def move_past_kv(past_kv: Any, device: torch.device | str) -> Any:
    """Move cache tensors; DynamicCache is mutated in place."""
    if past_kv is None:
        return None
    target = torch.device(device)
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            if torch.is_tensor(getattr(layer, "keys", None)):
                layer.keys = layer.keys.to(target)
            if torch.is_tensor(getattr(layer, "values", None)):
                layer.values = layer.values.to(target)
        return past_kv
    if isinstance(past_kv, (list, tuple)):
        return tuple(
            tuple(t.to(target) if torch.is_tensor(t) else t for t in layer)
            if isinstance(layer, (list, tuple))
            else (layer.to(target) if torch.is_tensor(layer) else layer)
            for layer in past_kv
        )
    return past_kv


def _tail(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    keep = min(max(tokens_to_keep, 0), tensor.shape[-2])
    return (
        tensor[..., tensor.shape[-2] - keep :, :].contiguous()
        if keep
        else tensor[..., 0:0, :].contiguous()
    )


def truncate_past_kv(past_kv: Any, tokens_to_keep: int) -> Any:
    """Keep the final cache positions; DynamicCache is mutated in place."""
    if past_kv is None or tokens_to_keep <= 0:
        return None
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            if torch.is_tensor(getattr(layer, "keys", None)):
                layer.keys = _tail(layer.keys, tokens_to_keep)
            if torch.is_tensor(getattr(layer, "values", None)):
                layer.values = _tail(layer.values, tokens_to_keep)
        return past_kv
    return tuple(
        tuple(_tail(t, tokens_to_keep) if torch.is_tensor(t) else t for t in layer)
        if isinstance(layer, (list, tuple))
        else (_tail(layer, tokens_to_keep) if torch.is_tensor(layer) else layer)
        for layer in past_kv
    )


def create_zero_past_kv(past_kv: Any) -> Any:
    cloned = clone_past_kv(past_kv)
    if cloned is None:
        return None
    if hasattr(cloned, "layers"):
        for tensor in _cache_tensors(cloned):
            tensor.zero_()
        return cloned
    return tuple(
        tuple(torch.zeros_like(t) if torch.is_tensor(t) else t for t in layer)
        if isinstance(layer, (list, tuple))
        else (torch.zeros_like(layer) if torch.is_tensor(layer) else layer)
        for layer in cloned
    )
