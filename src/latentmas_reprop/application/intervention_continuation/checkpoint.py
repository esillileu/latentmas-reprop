"""Prepare and resume reusable shared states."""

import copy
from pathlib import Path
from typing import Any

from ...domain.services.kv_cache import slice_past_kv
from ...domain.services.prompts import build_intervention_p1_messages
from ...infrastructure.storage.tensor_store import is_condition_complete, load_json


def _sample_identity(
    model: Any, args: Any, item: dict[str, Any], index: int
) -> dict[str, Any]:
    messages = build_intervention_p1_messages(item["question"], args)
    prompts, ids, _, _ = model.prepare_chat_batch(
        [messages], add_generation_prompt=True
    )
    return {
        "split": str(getattr(args, "split", "test")),
        "sample_index": index,
        "question": item["question"],
        "prompt": prompts[0],
        "prompt_token_ids": ids[0].cpu().tolist(),
    }


def _check_sample(meta: dict[str, Any], expected: dict[str, Any]) -> None:
    if any(meta.get(key) != value for key, value in expected.items()):
        raise ValueError("Stored sample identity differs from current input")


def _available_steps(state: Path, sample: str) -> list[int]:
    root = state / "samples" / sample / "shared"
    return sorted(
        (
            int(p.name[2:])
            for p in root.glob("ls*")
            if p.is_dir() and p.name[2:].isdigit()
        ),
        reverse=True,
    )


def prepare_state(
    storage: Any,
    runner: Any,
    state: Path,
    item: dict[str, Any],
    index: int,
    can_extend: bool,
    prepared: dict[int, Any] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    args, model = runner.args, runner.model

    def assemble(step: int) -> Any:
        if prepared is not None and step in prepared:
            result = prepared.pop(step)
            if isinstance(result, Exception):
                raise result
            return result
        return storage.assemble_shared_state(state, sample, step)

    sample, target = f"{index:06d}", int(getattr(args, "latent_steps", 10))
    expected = _sample_identity(model, args, item, index)
    steps = _available_steps(state, sample)
    base_dir = state / "samples" / sample / "shared" / "ls0"
    if steps and not is_condition_complete(base_dir):
        if not can_extend:
            raise ValueError(
                "Cannot repair shared state in a different execution environment"
            )
        marker = base_dir / "result.json"
        if marker.is_file():
            try:
                partial = load_json(marker)
            except (ValueError, OSError):
                partial = {}
            for key, value in expected.items():
                if key in partial and partial[key] != value:
                    raise ValueError(
                        "Stored sample identity differs from current input"
                    )
        for candidate in steps:
            if candidate == 0:
                continue
            directory = base_dir.parent / f"ls{candidate}"
            if is_condition_complete(directory):
                _, _, metadata = storage.load_shared_segment(state, sample, candidate)
                _check_sample(metadata, expected)
        prefill_runner = copy.copy(runner)
        prefill_runner.args = copy.copy(args)
        prefill_runner.args.latent_steps = 0

        def save_prefill(step: int, kv: Any, hidden: Any, meta: dict[str, Any]) -> None:
            if (
                meta.get("prompt") != expected["prompt"]
                or meta.get("prompt_token_ids") != expected["prompt_token_ids"]
            ):
                raise ValueError("Computed prompt differs from verified sample")
            storage.save_shared_segment(
                state, sample, 0, step, kv, hidden, dict(meta, **expected)
            )

        prefill_runner.compute_common_latent(item, checkpoint_callback=save_prefill)
        return prepare_state(storage, runner, state, item, index, can_extend, prepared)
    if (
        0 in steps
        and (state / "samples" / sample / "shared" / "ls0" / "result.json").is_file()
    ):
        _, _, base = storage.load_shared_segment(state, sample, 0)
        _check_sample(base, expected)
    else:
        base = None
    if base is not None:
        try:
            kv, hidden, meta = assemble(target)
        except ValueError:
            pass
        else:
            try:
                storage.load_shared_segment(state, sample, target)
                target_complete = True
            except (ValueError, KeyError):
                target_complete = False
            if not target_complete and can_extend:
                prior = 0
                for candidate in steps:
                    if candidate >= target:
                        continue
                    try:
                        storage.load_shared_segment(state, sample, candidate)
                        assemble(candidate)
                    except ValueError:
                        continue
                    prior = candidate
                    break
                prompt = int(base["prompt_length"])
                storage.save_shared_segment(
                    state,
                    sample,
                    prior,
                    target,
                    slice_past_kv(kv, prompt + prior, prompt + target),
                    hidden,
                    dict(meta, **expected, latent_steps=target),
                )
            return kv, hidden, dict(base, latent_steps=target)
    if not can_extend:
        raise ValueError(
            "Cannot extend shared state in a different execution environment"
        )
    initial_step, initial_kv, initial_hidden, initial_meta = 0, None, None, None
    for candidate in steps:
        if candidate >= target:
            continue
        try:
            storage.load_shared_segment(state, sample, candidate)
            initial_kv, initial_hidden, initial_meta = assemble(candidate)
        except ValueError:
            continue
        initial_step = candidate
        break
    previous = initial_step

    def save(step: int, kv: Any, hidden: Any, meta: dict[str, Any]) -> None:
        nonlocal previous
        if (
            meta.get("prompt") != expected["prompt"]
            or meta.get("prompt_token_ids") != expected["prompt_token_ids"]
        ):
            raise ValueError("Computed prompt differs from verified sample")
        prompt = int(meta["prompt_length"])
        segment = slice_past_kv(
            kv, 0 if step == 0 else prompt + previous, prompt + step
        )
        storage.save_shared_segment(
            state, sample, previous, step, segment, hidden, dict(meta, **expected)
        )
        previous = step

    kv, hidden, meta = runner.compute_common_latent(
        dict(item, **(initial_meta or {})),
        initial_kv,
        initial_hidden,
        initial_step,
        save,
    )
    if target != 0:
        save(target, kv, hidden, meta)
    return kv, hidden, dict(meta, **expected)
