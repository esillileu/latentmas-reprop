"""Tests for KV cache operations, zero cache semantics, and DynamicCache truncation."""

import torch
from transformers.cache_utils import DynamicCache

from latentmas_reprop.domain.services.kv_cache import (
    clone_past_kv,
    create_zero_past_kv,
    estimate_past_kv_bytes,
    get_past_kv_num_layers,
    get_past_kv_sequence_length,
    move_past_kv,
    truncate_past_kv,
)


def _make_cache(seq_len: int = 8) -> DynamicCache:
    cache = DynamicCache()
    k = torch.randn(1, 2, seq_len, 16)
    v = torch.randn(1, 2, seq_len, 16)
    cache.update(k, v, 0)
    return cache


def test_drop_is_none():
    """Drop condition => no past_key_values (None), not zero cache."""
    drop_ctx = None
    assert drop_ctx is None


def test_zero_is_same_shape_all_zeros():
    """Zero condition => same shape as own cache, all tensor values == 0."""
    cache = _make_cache(seq_len=8)
    zeroed = create_zero_past_kv(cache)

    assert zeroed.layers[0].keys.shape == cache.layers[0].keys.shape
    assert zeroed.layers[0].values.shape == cache.layers[0].values.shape
    assert zeroed.layers[0].keys.abs().sum().item() == 0.0
    assert zeroed.layers[0].values.abs().sum().item() == 0.0
    assert cache.layers[0].keys.abs().sum().item() > 0.0


def test_drop_ne_zero_semantically():
    """Drop (None) and zero (zero-filled cache) are structurally different."""
    cache = _make_cache(seq_len=8)
    drop_ctx = None
    zero_ctx = create_zero_past_kv(cache)
    assert drop_ctx is None
    assert zero_ctx is not None
    assert get_past_kv_sequence_length(drop_ctx) == 0
    assert get_past_kv_sequence_length(zero_ctx) == 8


def test_cache_helpers_dynamic_cache():
    cache = DynamicCache()
    k = torch.ones(1, 2, 4, 8)
    v = torch.ones(1, 2, 4, 8) * 2
    cache.update(k, v, 0)

    cloned = clone_past_kv(cache)
    assert cloned is not cache
    assert torch.equal(cloned.layers[0].keys, cache.layers[0].keys)

    moved = move_past_kv(cloned, "cpu")
    assert moved.layers[0].keys.device.type == "cpu"

    zeroed = create_zero_past_kv(cache)
    assert torch.all(zeroed.layers[0].keys == 0)
    assert torch.all(zeroed.layers[0].values == 0)
    assert zeroed.layers[0].keys.shape == k.shape


def test_get_past_kv_sequence_length():
    assert get_past_kv_sequence_length(None) == 0
    cache = _make_cache(seq_len=7)
    assert get_past_kv_sequence_length(cache) == 7


def test_get_past_kv_num_layers():
    assert get_past_kv_num_layers(None) == 0
    cache = DynamicCache()
    cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), 0)
    cache.update(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8), 1)
    assert get_past_kv_num_layers(cache) == 2


def test_estimate_past_kv_bytes():
    cache = _make_cache(seq_len=8)
    expected = 2 * (1 * 2 * 8 * 16 * 4)
    actual = estimate_past_kv_bytes(cache)
    assert actual == expected


def test_cache_length_metadata_own():
    cache = _make_cache(seq_len=12)
    seq_len = get_past_kv_sequence_length(cache)
    assert seq_len == 12


def test_cache_length_metadata_cross_delta():
    cache_i = _make_cache(seq_len=10)
    cache_j = _make_cache(seq_len=15)
    target = get_past_kv_sequence_length(cache_i)
    src = get_past_kv_sequence_length(cache_j)
    assert src - target == 5


def test_cache_length_metadata_drop():
    assert get_past_kv_sequence_length(None) == 0


def test_truncate_past_requested_vs_actual():
    """Truncation results in exactly the requested number of tokens."""
    cache = DynamicCache()
    k = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    v = torch.arange(20, dtype=torch.float32).view(1, 1, 20, 1)
    cache.update(k, v, 0)
    assert get_past_kv_sequence_length(cache) == 20

    truncated = truncate_past_kv(cache, tokens_to_keep=7)
    actual_len = get_past_kv_sequence_length(truncated)
    assert actual_len == 7
    assert truncated.layers[0].keys.squeeze()[-1].item() == 19.0


def test_truncate_past_keeps_max_when_small():
    """If tokens_to_keep >= cache length, entire cache is kept."""
    cache = _make_cache(seq_len=5)
    result = truncate_past_kv(cache, tokens_to_keep=100)
    assert get_past_kv_sequence_length(result) == 5
