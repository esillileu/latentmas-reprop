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
    if past_kv is None:
        return None
    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            if torch.is_tensor(getattr(layer, "keys", None)) and not layer.keys.is_leaf:
                layer.keys = layer.keys.detach()
            if (
                torch.is_tensor(getattr(layer, "values", None))
                and not layer.values.is_leaf
            ):
                layer.values = layer.values.detach()
    elif isinstance(past_kv, (list, tuple)):
        past_kv = tuple(
            tuple(
                t.detach() if torch.is_tensor(t) and not t.is_leaf else t for t in layer
            )
            if isinstance(layer, (list, tuple))
            else (
                layer.detach()
                if torch.is_tensor(layer) and not layer.is_leaf
                else layer
            )
            for layer in past_kv
        )
    return copy.deepcopy(past_kv)


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


def retain_past_kv_prefix(past_kv: Any, tokens_to_keep: int) -> Any:
    """Keep the first cache positions; DynamicCache is mutated in place."""
    if past_kv is None or tokens_to_keep <= 0:
        return None

    def prefix(tensor: torch.Tensor) -> torch.Tensor:
        keep = min(tokens_to_keep, tensor.shape[-2])
        return tensor[..., :keep, :].contiguous()

    if hasattr(past_kv, "layers"):
        for layer in past_kv.layers:
            if torch.is_tensor(getattr(layer, "keys", None)):
                layer.keys = prefix(layer.keys)
            if torch.is_tensor(getattr(layer, "values", None)):
                layer.values = prefix(layer.values)
        return past_kv
    return tuple(
        tuple(prefix(t) if torch.is_tensor(t) else t for t in layer)
        if isinstance(layer, (list, tuple))
        else (prefix(layer) if torch.is_tensor(layer) else layer)
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


def _dense_layers(cache: Any) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], bool]:
    """Extract only cache types whose complete state is represented by dense K/V."""
    from transformers.cache_utils import DynamicCache, DynamicLayer

    dynamic = type(cache) is DynamicCache
    if dynamic:
        if getattr(cache, "offloading", False) or any(
            type(layer) is not DynamicLayer for layer in cache.layers
        ):
            raise ValueError("Unsupported DynamicCache layer or offloading")
        layers = [(layer.keys, layer.values) for layer in cache.layers]
    elif isinstance(cache, tuple):
        layers = list(cache)
    else:
        raise ValueError("Unsupported KV cache type")
    if not layers:
        raise ValueError("Empty KV cache")
    length = None
    for layer in layers:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise ValueError("Expected key and value per layer")
        for tensor in layer:
            if not torch.is_tensor(tensor) or tensor.ndim < 2:
                raise ValueError("Expected dense KV tensors")
            if length is None:
                length = tensor.shape[-2]
            if tensor.shape[-2] != length:
                raise ValueError("Inconsistent KV sequence lengths")
    return layers, dynamic


def _build_dense_cache(
    layers: list[tuple[torch.Tensor, torch.Tensor]], dynamic: bool
) -> Any:
    if dynamic:
        from transformers.cache_utils import DynamicCache

        return DynamicCache(ddp_cache_data=layers)
    return tuple(layers)


def slice_past_kv(cache: Any, start: int, end: int) -> Any:
    """Return an independent CPU slice on the sequence axis."""
    layers, dynamic = _dense_layers(cache)
    length = layers[0][0].shape[-2]
    if not 0 <= start <= end <= length:
        raise ValueError("Invalid KV slice bounds")
    return _build_dense_cache(
        [
            (
                k[..., start:end, :].detach().cpu().clone(),
                v[..., start:end, :].detach().cpu().clone(),
            )
            for k, v in layers
        ],
        dynamic,
    )


def concat_past_kv(segments: list[Any]) -> Any:
    """Concatenate compatible dense cache segments without mutating them."""
    if not segments:
        raise ValueError("No KV segments")
    extracted = [_dense_layers(segment) for segment in segments]
    first, dynamic = extracted[0]
    for layers, kind in extracted[1:]:
        if kind != dynamic or len(layers) != len(first):
            raise ValueError("Incompatible KV cache types or layer counts")
        for reference, current in zip(first, layers, strict=True):
            for a, b in zip(reference, current, strict=True):
                if (
                    a.dtype != b.dtype
                    or a.shape[:-2] + a.shape[-1:] != b.shape[:-2] + b.shape[-1:]
                ):
                    raise ValueError("Incompatible KV tensor shape or dtype")
    return _build_dense_cache(
        [
            tuple(
                torch.cat(
                    [layers[i][j].detach().cpu() for layers, _ in extracted], dim=-2
                )
                for j in range(2)
            )
            for i in range(len(first))
        ],
        dynamic,
    )
