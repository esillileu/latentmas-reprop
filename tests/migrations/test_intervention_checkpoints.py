"""Legacy cumulative KV conversion checks."""

import pytest
import torch

from latentmas_reprop.domain.services.kv_cache import concat_past_kv, slice_past_kv
from tools.migrations.intervention_continuation.checkpoints import (
    Checkpoint,
    equal_kv,
    split_checkpoints,
)


def checkpoint(step: int) -> Checkpoint:
    prompt = 3
    values = torch.arange((prompt + 23) * 2).reshape(1, 1, prompt + 23, 2)
    kv = (
        (values[:, :, : prompt + step].clone(), values[:, :, : prompt + step].clone()),
    )
    hidden = torch.arange((prompt + step) * 2).reshape(prompt + step, 2)
    return Checkpoint(
        step,
        kv,
        hidden,
        {
            "sample_index": 0,
            "prompt_length": prompt,
            "prompt_token_ids": [1, 2, 3],
            "prompt": "prompt",
            "latent_steps": step,
        },
    )


def test_split_and_validate_partial_boundaries():
    checkpoints = [checkpoint(15), checkpoint(23), checkpoint(10)]
    parts = split_checkpoints(checkpoints)
    assert [part.step for part in parts] == [0, 10, 15, 20, 23]
    assert [part.kv[0][0].shape[-2] for part in parts] == [3, 10, 5, 5, 3]
    assert equal_kv(parts[0].kv, slice_past_kv(checkpoints[0].kv, 0, 3))


def test_rejects_inconsistent_prefix_and_hidden():
    first, second = checkpoint(10), checkpoint(20)
    first.kv[0][0][..., 0, 0] += 1
    with pytest.raises(ValueError, match="KV prefix"):
        split_checkpoints([first, second])
    first = checkpoint(10)
    first.hidden[0, 0] += 1
    with pytest.raises(ValueError, match="hidden prefix"):
        split_checkpoints([first, second])


def test_missing_prompt_identity_is_allowed():
    item = checkpoint(10)
    item.metadata["prompt_token_ids"] = []
    assert split_checkpoints([item])[-1].step == 10


def test_dynamic_cache_segments_assemble_exactly():
    from transformers.cache_utils import DynamicCache

    item = checkpoint(15)
    original = DynamicCache(ddp_cache_data=list(item.kv))
    dynamic = Checkpoint(item.step, original, item.hidden, item.metadata)
    parts = split_checkpoints([dynamic])
    assert equal_kv(concat_past_kv([part.kv for part in parts]), original)
    assert original.get_seq_length() == 18
    assert [part.step for part in split_checkpoints([checkpoint(10), dynamic])] == [
        0,
        10,
        15,
    ]
