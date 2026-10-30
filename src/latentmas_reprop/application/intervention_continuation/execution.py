"""Persist and reuse intervention continuation states and condition results."""

from pathlib import Path
from typing import Any

import torch

from ...domain.services.kv_cache import slice_past_kv
from ...domain.services.prompts import build_intervention_p1_messages
from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .identity import (
    canonical_hash,
    ensure_state,
    result_identity,
    state_identity,
    state_path,
)


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
) -> tuple[Any, Any, dict[str, Any]]:
    args, model = runner.args, runner.model
    sample, target = f"{index:06d}", int(getattr(args, "latent_steps", 10))
    expected = _sample_identity(model, args, item, index)
    steps = _available_steps(state, sample)
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
            kv, hidden, meta = storage.assemble_shared_state(state, sample, target)
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
                        storage.assemble_shared_state(state, sample, candidate)
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
            initial_kv, initial_hidden, initial_meta = storage.assemble_shared_state(
                state, sample, candidate
            )
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


def execute(
    use_case: Any, model: Any, args: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from ...infrastructure.paths.resolver import get_git_commit_hash
    from .artifacts import (
        build_summary_dict,
        log_to_mlflow_tracker,
        save_experiment_summary,
        start_intervention_continuation_tracker,
        trace_intervention_continuation_sample,
    )
    from .runner import InterventionContinuationRunner

    if getattr(args, "overwrite", False):
        raise ValueError(
            "Overwriting shared states is not supported; choose a separate storage root"
        )
    backend = "vllm" if getattr(args, "use_vllm", False) else "transformers"
    identity, provenance = state_identity(model, args, backend)
    state = state_path(use_case.storage.base_dir, identity)
    can_extend = ensure_state(state, identity, provenance)
    dataset = list(
        use_case.dataset_port.load(task=identity["task"], split=identity["split"])
    )
    limit = int(getattr(args, "max_samples", 0))
    if limit > 0:
        dataset = dataset[:limit]
    raw = getattr(args, "intervention_conditions", None) or ["ple", "plce", "plpe"]
    conditions = (
        [c.strip() for c in raw.split(",")] if isinstance(raw, str) else list(raw)
    )
    if set(conditions) - {"ple", "plce", "plpe"}:
        raise ValueError("Unknown intervention condition")
    step = int(getattr(args, "latent_steps", 10))
    runner = InterventionContinuationRunner(model, args, use_case.evaluator_port)
    records = []
    paths = {condition: [] for condition in conditions}
    for index, item in enumerate(dataset):
        sample = f"{index:06d}"
        kv, hidden, _ = prepare_state(
            use_case.storage, runner, state, item, index, can_extend
        )
        for condition in conditions:
            ident = result_identity(state.name, step, condition, item, args, model)
            result_hash = canonical_hash(ident)
            directory = (
                state
                / "samples"
                / sample
                / "results"
                / f"ls{step}"
                / result_hash
                / condition
            )
            paths[condition].append(str(directory.relative_to(state)))
            if is_condition_complete(directory):
                _, record = use_case.storage.load_intervention_result(directory, ident)
            else:
                rng = {
                    "cpu": torch.get_rng_state().tolist(),
                    "order": {
                        "sample_index": index,
                        "condition": condition,
                        "conditions": conditions,
                    },
                }
                if torch.device(model.device).type == "cuda":
                    rng["cuda"] = torch.cuda.get_rng_state(model.device).tolist()
                if condition == "ple":
                    branch, record = runner.execute_ple(item, kv, hidden[-1])
                else:
                    branch, record = getattr(runner, f"execute_{condition}")(item, kv)
                record = dict(
                    record,
                    status="complete",
                    sample_index=index,
                    identity=ident,
                    files=["hidden.pt"],
                    top_p_applied=False,
                    rng_provenance=rng,
                )
                use_case.storage.save_intervention_result(
                    directory, ident, record, branch
                )
                if (
                    record.get("error") is not None
                    or record.get("error_msg") is not None
                ):
                    raise ValueError(
                        f"Condition {condition} failed: {record.get('error_msg') or record.get('error')}"
                    )
            records.append(record)
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        run_id = canonical_hash(
            {
                "state": state.name,
                "step": step,
                "condition": condition,
                "results": paths[condition],
            }
        )
        run_dir = state / "runs" / run_id
        new_run = not (run_dir / "summary.json").is_file()
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "state_path": str(state.relative_to(use_case.storage.base_dir)),
            "step": step,
            "condition": condition,
            "result_paths": paths[condition],
        }
        config_path = run_dir / "config.json"
        if config_path.is_file() and load_json(config_path) != config:
            raise ValueError("Run identity mismatch")
        if not config_path.is_file():
            save_json_atomic(config_path, config)
        summary = build_summary_dict(subset, [condition], len(dataset), config)
        save_experiment_summary(run_dir, summary, subset)
        if new_run and use_case.tracker_port:
            relative = str(run_dir.relative_to(use_case.storage.base_dir))
            git_hash = get_git_commit_hash()
            started = start_intervention_continuation_tracker(
                use_case.tracker_port, run_dir, relative, git_hash, args
            )
            for index, (item, record) in enumerate(zip(dataset, subset, strict=True)):
                _, _, meta = use_case.storage.assemble_shared_state(
                    state, f"{index:06d}", step
                )
                trace_intervention_continuation_sample(
                    use_case.tracker_port,
                    args,
                    index,
                    f"{index:06d}",
                    item,
                    meta,
                    {condition: record},
                    git_hash,
                )
            log_to_mlflow_tracker(
                use_case.tracker_port,
                run_dir,
                relative,
                git_hash,
                args,
                summary,
                run_already_started=started,
            )
    return build_summary_dict(records, conditions, len(dataset), vars(args)), records
