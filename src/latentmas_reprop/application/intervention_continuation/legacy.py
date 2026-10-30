"""Reuse migrated checkpoints after comparing them with the loaded model."""

import copy
from pathlib import Path
from typing import Any

import torch

from ...domain.services.kv_cache import _build_dense_cache, _dense_layers, slice_past_kv
from ...infrastructure.storage.tensor_store import is_condition_complete, load_json
from .determinism import StochasticForwardError


def _equal_kv(left: Any, right: Any) -> bool:
    a, _ = _dense_layers(left)
    b, _ = _dense_layers(right)
    return len(a) == len(b) and all(
        x.dtype == y.dtype and torch.equal(x, y)
        for la, lb in zip(a, b, strict=True)
        for x, y in zip(la, lb, strict=True)
    )


def reuse_migrated_state(
    storage: Any,
    runner: Any,
    state: Path,
    item: dict[str, Any],
    sample: str,
    target: int,
    expected: dict[str, Any],
) -> bool:
    """Check the prompt and final transition, then publish an existing KV prefix."""
    current_step = 0
    shared_current = state / "samples" / sample / "shared"
    if is_condition_complete(shared_current / "ls0"):
        steps = sorted(
            (
                int(path.name[2:])
                for path in shared_current.glob("ls*")
                if path.is_dir()
                and path.name[2:].isdigit()
                and 0 < int(path.name[2:]) < target
            ),
            reverse=True,
        )
        for step in steps:
            try:
                storage.assemble_shared_state(state, sample, step)
            except ValueError:
                continue
            current_step = step
            break
    candidates = []
    for source in sorted(state.parent.iterdir()):
        if source == state or not (source / "state.json").is_file():
            continue
        identity = load_json(source / "state.json").get("identity", {})
        if "legacy_group" not in identity:
            continue
        shared = source / "samples" / sample / "shared"
        if not is_condition_complete(shared / "ls0"):
            continue
        meta = load_json(shared / "ls0" / "result.json")
        if any(
            meta.get(key) is not None and meta[key] != value
            for key, value in expected.items()
        ):
            continue
        steps = [
            int(path.name[2:])
            for path in shared.glob("ls*")
            if path.is_dir() and path.name[2:].isdigit() and is_condition_complete(path)
        ]
        if steps and min(max(steps), target) > current_step:
            candidates.append((min(max(steps), target), source))
    if not candidates:
        return False

    base = state / "samples" / sample / "shared" / "ls0"
    if is_condition_complete(base):
        current_kv, current_hidden, current_meta = storage.assemble_shared_state(
            state, sample, 0
        )
    else:
        prefill = copy.copy(runner)
        prefill.args = copy.copy(runner.args)
        prefill.args.latent_steps = 0
        current_kv, current_hidden, current_meta = prefill.compute_common_latent(item)
        storage.save_shared_segment(
            state,
            sample,
            0,
            0,
            current_kv,
            current_hidden,
            dict(current_meta, **expected),
        )
    for step, source in sorted(
        candidates, key=lambda candidate: candidate[0], reverse=True
    ):
        try:
            old_kv, old_hidden, _ = storage.assemble_shared_state(source, sample, 0)
            legacy_kv, legacy_hidden, _ = storage.assemble_shared_state(
                source, sample, step
            )
            if (
                not _equal_kv(current_kv, old_kv)
                or current_hidden.dtype != old_hidden.dtype
                or not torch.equal(current_hidden, old_hidden)
            ):
                continue
            if current_step:
                current_prefix, current_trajectory, _ = storage.assemble_shared_state(
                    state, sample, current_step
                )
                legacy_prefix, legacy_trajectory, _ = storage.assemble_shared_state(
                    source, sample, current_step
                )
                if (
                    not _equal_kv(current_prefix, legacy_prefix)
                    or current_trajectory.dtype != legacy_trajectory.dtype
                    or not torch.equal(current_trajectory, legacy_trajectory)
                ):
                    continue
            if step:
                previous_kv, previous_hidden, _ = storage.assemble_shared_state(
                    source, sample, step - 1
                )
                _, dynamic = _dense_layers(current_kv)
                previous_kv = _build_dense_cache(_dense_layers(previous_kv)[0], dynamic)
                probe = copy.copy(runner)
                probe.args = copy.copy(runner.args)
                probe.args.latent_steps = step
                actual_kv, actual_hidden, _ = probe.compute_common_latent(
                    dict(item, **current_meta),
                    initial_kv=previous_kv,
                    initial_hiddens=previous_hidden,
                    initial_step=step - 1,
                )
                if (
                    not _equal_kv(actual_kv, legacy_kv)
                    or actual_hidden.dtype != legacy_hidden.dtype
                    or not torch.equal(actual_hidden, legacy_hidden)
                ):
                    continue
                prompt_length = int(current_meta["prompt_length"])
                storage.save_shared_segment(
                    state,
                    sample,
                    current_step,
                    step,
                    _build_dense_cache(
                        _dense_layers(
                            slice_past_kv(
                                legacy_kv,
                                prompt_length + current_step,
                                prompt_length + step,
                            )
                        )[0],
                        dynamic,
                    ),
                    legacy_hidden,
                    dict(current_meta, **expected, latent_steps=step),
                )
            return True
        except StochasticForwardError:
            raise
        except (ValueError, OSError, KeyError):
            continue
    return False
