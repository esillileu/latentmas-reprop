"""Immutable intervention continuation shared-state segments."""

from pathlib import Path
from typing import Any

import torch


def save_shared_segment(
    self,
    state: Path | str,
    sample: str,
    start: int,
    end: int,
    kv: Any,
    hidden: torch.Tensor,
    metadata: dict[str, Any],
) -> Path:
    """Commit an immutable segment; hidden contains the full trajectory."""
    from ...domain.services.kv_cache import (
        _dense_layers,
        get_past_kv_sequence_length,
        slice_past_kv,
    )
    from .tensor_store import is_condition_complete, save_condition_result

    if start < 0 or end < start or (end == 0 and start != 0):
        raise ValueError("Invalid segment bounds")
    prompt = int(metadata["prompt_length"])
    expected = prompt if end == 0 else end - start
    layers, dynamic = _dense_layers(kv)
    if (
        expected <= 0
        or get_past_kv_sequence_length(kv) != expected
        or hidden.shape[0] != prompt + end
    ):
        raise ValueError("Invalid segment or hidden length")
    kv = slice_past_kv(kv, 0, expected)
    hidden = hidden.detach().cpu().clone()
    directory = Path(state) / "samples" / sample / "shared" / f"ls{end}"
    if is_condition_complete(directory):
        try:
            self.load_shared_segment(state, sample, end)
        except (ValueError, KeyError, TypeError):
            (directory / "result.json").unlink()
        else:
            raise ValueError(f"Shared segment already complete: {directory}")
    # An interrupted write must not leave a completion marker visible while replacing tensors.
    (directory / "result.json").unlink(missing_ok=True)
    payload = dict(
        metadata,
        status="complete",
        start_step=start,
        end_step=end,
        cache_type="dynamic" if dynamic else "tuple",
        layer_count=len(layers),
        tensors=[
            [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in layer]
            for layer in layers
        ],
        files=["kv.pt", "hidden.pt"],
    )
    return save_condition_result(directory, payload, {"kv": kv, "hidden": hidden})


def load_shared_segment(
    self, state: Path | str, sample: str, step: int
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    from .tensor_store import is_condition_complete, load_json, load_tensor

    directory = Path(state) / "samples" / sample / "shared" / f"ls{step}"
    if not is_condition_complete(directory):
        raise ValueError(f"Incomplete shared segment: {directory}")
    meta = load_json(directory / "result.json")
    if (
        meta.get("status") != "complete"
        or set(meta.get("files", [])) != {"kv.pt", "hidden.pt"}
        or meta.get("end_step") != step
    ):
        raise ValueError(f"Invalid shared segment metadata: {directory}")
    kv = load_tensor(directory / "kv.pt", map_location="cpu")
    hidden = load_tensor(directory / "hidden.pt", map_location="cpu")
    from ...domain.services.kv_cache import _dense_layers

    layers, dynamic = _dense_layers(kv)
    specs = [
        [{"shape": list(t.shape), "dtype": str(t.dtype)} for t in layer]
        for layer in layers
    ]
    start, prompt = int(meta["start_step"]), int(meta["prompt_length"])
    expected = prompt if step == 0 and start == 0 else step - start
    if (
        start < 0
        or start > step
        or expected <= 0
        or meta.get("cache_type") != ("dynamic" if dynamic else "tuple")
        or meta.get("layer_count") != len(layers)
        or meta.get("tensors") != specs
        or any(t.shape[-2] != expected for layer in layers for t in layer)
        or not torch.is_tensor(hidden)
        or hidden.shape[0] != prompt + step
    ):
        raise ValueError(f"Invalid shared segment contents: {directory}")
    return kv, hidden, meta


def assemble_shared_state(
    self, state: Path | str, sample: str, step: int
) -> tuple[Any, torch.Tensor, dict[str, Any]]:
    """Assemble an exact prefix, including from a segment ending after step."""
    from ...domain.services.kv_cache import concat_past_kv, slice_past_kv

    if step < 0:
        raise ValueError("Negative step")
    kv, hidden, meta = self.load_shared_segment(state, sample, 0)
    segments = [kv]
    prompt = int(meta["prompt_length"])
    base_meta = meta
    current = 0
    directory = Path(state) / "samples" / sample / "shared"
    candidates = sorted(
        int(p.name[2:])
        for p in directory.glob("ls*")
        if p.is_dir() and p.name[2:].isdigit() and int(p.name[2:]) > 0
    )
    while current < step:
        choices = []
        for end in candidates:
            if end <= current:
                continue
            from .tensor_store import is_condition_complete

            if not is_condition_complete(directory / f"ls{end}"):
                continue
            part, part_hidden, part_meta = self.load_shared_segment(state, sample, end)
            if (
                part_meta["start_step"] <= current
                and part_meta["prompt_length"] == prompt
            ):
                choices.append((end, part, part_hidden, part_meta))
        if not choices:
            raise ValueError(
                f"Missing or discontinuous shared segment after ls{current}"
            )
        end, part, next_hidden, meta = min(choices, key=lambda x: x[0])
        if not torch.equal(hidden[: prompt + current], next_hidden[: prompt + current]):
            raise ValueError("Shared hidden trajectories disagree")
        hidden = next_hidden
        if any(
            meta.get(key) != base_meta.get(key)
            for key in (
                "split",
                "sample_index",
                "question",
                "prompt",
                "prompt_token_ids",
            )
        ):
            raise ValueError("Shared segment sample identity mismatch")
        take = min(end, step) - current
        offset = current - int(meta["start_step"])
        if offset:
            from ...domain.services.kv_cache import _dense_layers

            existing = slice_past_kv(
                concat_past_kv(segments), prompt + current - offset, prompt + current
            )
            overlap = slice_past_kv(part, 0, offset)
            left, _ = _dense_layers(existing)
            right, _ = _dense_layers(overlap)
            if len(left) != len(right) or any(
                not torch.equal(a, b)
                for x, y in zip(left, right, strict=True)
                for a, b in zip(x, y, strict=True)
            ):
                raise ValueError("Overlapping shared segments disagree")
        segments.append(slice_past_kv(part, offset, offset + take))
        current += take
    return concat_past_kv(segments), hidden[: prompt + step].clone(), meta


def save_intervention_result(
    self,
    directory: Path | str,
    identity: dict[str, Any],
    record: dict[str, Any],
    hidden: torch.Tensor,
) -> Path:
    """Commit a condition result after its tensor; completed results are immutable."""
    from .tensor_store import is_condition_complete, save_condition_result

    directory = Path(directory)
    if (directory / "result.json").is_file():
        if is_condition_complete(directory):
            raise ValueError(f"Result already complete: {directory}")
        (directory / "result.json").unlink()
    payload = dict(record, identity=identity, status="complete", files=["hidden.pt"])
    return save_condition_result(directory, payload, {"hidden": hidden})


def load_intervention_result(
    self, directory: Path | str, identity: dict[str, Any] | None = None
) -> tuple[torch.Tensor, dict[str, Any]]:
    from .tensor_store import is_condition_complete, load_json, load_tensor

    directory = Path(directory)
    if not is_condition_complete(directory):
        raise ValueError(f"Incomplete result: {directory}")
    record = load_json(directory / "result.json")
    if (
        record.get("status") != "complete"
        or record.get("files") != ["hidden.pt"]
        or (identity is not None and record.get("identity") != identity)
    ):
        raise ValueError(f"Invalid result: {directory}")
    hidden = load_tensor(directory / "hidden.pt", map_location="cpu")
    if not torch.is_tensor(hidden):
        raise ValueError(f"Invalid result tensor: {directory}")
    return hidden, record
