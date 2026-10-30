import pytest
import torch

from latentmas_reprop.domain.services.kv_cache import concat_past_kv, slice_past_kv
from latentmas_reprop.infrastructure.storage.tensor_store import FileStorage


def test_segment_round_trip(tmp_path):
    original = tuple(
        (torch.randn(1, 2, 23, 4), torch.randn(1, 2, 23, 4)) for _ in range(2)
    )
    parts = [
        slice_past_kv(original, 0, 3),
        slice_past_kv(original, 3, 13),
        slice_past_kv(original, 13, 23),
    ]
    rebuilt = concat_past_kv(parts)
    for source, result in zip(original, rebuilt, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(source, result, strict=True))
    store = FileStorage()
    hidden = torch.randn(23, 4)
    for start, end, part in [(0, 0, parts[0]), (0, 10, parts[1]), (10, 20, parts[2])]:
        store.save_shared_segment(
            tmp_path,
            "000000",
            start,
            end,
            part,
            hidden[: 3 + end],
            {"prompt_length": 3},
        )
    for step in (0, 10, 15, 20):
        kv, trajectory, _ = store.assemble_shared_state(tmp_path, "000000", step)
        assert torch.equal(trajectory, hidden[: 3 + step])
        for source, result in zip(original, kv, strict=True):
            assert all(
                torch.equal(a[..., : 3 + step, :], b)
                for a, b in zip(source, result, strict=True)
            )
    with pytest.raises(ValueError):
        store.save_shared_segment(
            tmp_path, "000000", 0, 10, parts[1], hidden[:13], {"prompt_length": 3}
        )


def test_dynamic_cache_slices_are_independent():
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    for layer in range(2):
        cache.update(torch.randn(1, 2, 13, 4), torch.randn(1, 2, 13, 4), layer)
    first = slice_past_kv(cache, 0, 3)
    second = slice_past_kv(cache, 3, 13)
    joined = concat_past_kv([first, second])
    assert joined.get_seq_length() == 13
    for expected, actual in zip(cache.layers, joined.layers, strict=True):
        assert torch.equal(expected.keys, actual.keys)
        assert torch.equal(expected.values, actual.values)
    first.layers[0].keys.zero_()
    assert not torch.equal(first.layers[0].keys, cache.layers[0].keys[..., :3, :])


def test_missing_and_partial_segment_is_not_complete(tmp_path):
    store = FileStorage()
    cache = ((torch.ones(1, 1, 3, 2), torch.ones(1, 1, 3, 2)),)
    hidden = torch.ones(3, 2)
    store.save_shared_segment(
        tmp_path, "000000", 0, 0, cache, hidden, {"prompt_length": 3}
    )
    directory = tmp_path / "samples/000000/shared/ls10"
    directory.mkdir(parents=True)
    (directory / "result.json").write_text(
        '{"status":"complete","files":["kv.pt","hidden.pt"]}', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Missing or discontinuous"):
        store.assemble_shared_state(tmp_path, "000000", 10)


def test_conflicting_overlap_is_rejected(tmp_path):
    store = FileStorage()

    def cache(length, value):
        tensor = torch.full((1, 1, length, 1), value)
        return ((tensor, tensor.clone()),)

    meta = {"prompt_length": 3}
    store.save_shared_segment(
        tmp_path, "000000", 0, 0, cache(3, 1), torch.ones(3, 1), meta
    )
    store.save_shared_segment(
        tmp_path, "000000", 0, 15, cache(15, 1), torch.ones(18, 1), meta
    )
    store.save_shared_segment(
        tmp_path, "000000", 10, 20, cache(10, 2), torch.ones(23, 1), meta
    )
    with pytest.raises(ValueError, match="Overlapping"):
        store.assemble_shared_state(tmp_path, "000000", 20)
