"""Persist and reuse intervention continuation states and condition results."""

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .checkpoint import _available_steps, _check_sample, _sample_identity, prepare_state
from .determinism import StochasticForwardError
from .identity import (
    canonical_hash,
    ensure_state,
    result_identity,
    state_identity,
    state_path,
)


def _prepare_sample(
    storage: Any, state: Any, sample: str, steps: set[int]
) -> dict[int, Any]:
    """Load only CPU states; unavailable states follow the normal resume path."""
    if not is_condition_complete(state / "samples" / sample / "shared" / "ls0"):
        return {}
    prepared = {}
    for step in steps:
        try:
            prepared[step] = storage.assemble_shared_state(state, sample, step)
        except ValueError as exc:
            prepared[step] = exc
            for candidate in _available_steps(state, sample):
                if candidate >= step or candidate in prepared:
                    continue
                try:
                    storage.load_shared_segment(state, sample, candidate)
                    prepared[candidate] = storage.assemble_shared_state(
                        state, sample, candidate
                    )
                except ValueError:
                    continue
                break
    return prepared


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
    invalid = state / "invalid.json"
    if invalid.exists():
        raise ValueError("Shared state was invalidated by stochastic forward")
    dataset = list(
        use_case.dataset_port.load(task=identity["task"], split=identity["split"])
    )
    limit = int(getattr(args, "max_samples", 0))
    if limit > 0:
        dataset = dataset[:limit]
    raw = getattr(args, "intervention_conditions", None) or [
        "pe",
        "ple",
        "plce",
        "plpe",
    ]
    conditions = (
        [c.strip() for c in raw.split(",")] if isinstance(raw, str) else list(raw)
    )
    if set(conditions) - {"pe", "ple", "plce", "plpe"}:
        raise ValueError("Unknown intervention condition")
    step = int(getattr(args, "latent_steps", 10))
    runner = InterventionContinuationRunner(model, args, use_case.evaluator_port)
    records = []
    paths = {condition: [] for condition in conditions}
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for index, item in enumerate(dataset):
            prepared = future.result() if future is not None else {}
            future = None
            if index + 1 < len(dataset):
                next_index = index + 1
                next_item = dataset[next_index]
                next_sample = f"{next_index:06d}"
                needed = set()
                for condition in conditions:
                    condition_step = 0 if condition == "pe" else step
                    ident = result_identity(
                        state.name, condition_step, condition, next_item, args, model
                    )
                    directory = (
                        state
                        / "samples"
                        / next_sample
                        / "results"
                        / f"ls{condition_step}"
                        / canonical_hash(ident)
                        / condition
                    )
                    if not is_condition_complete(directory):
                        needed.add(condition_step)
                if needed:
                    future = executor.submit(
                        _prepare_sample, use_case.storage, state, next_sample, needed
                    )
            sample = f"{index:06d}"
            base = state / "samples" / sample / "shared" / "ls0"
            if is_condition_complete(base):
                _, _, meta = use_case.storage.load_shared_segment(state, sample, 0)
                _check_sample(meta, _sample_identity(model, args, item, index))
            states = {}
            if not is_condition_complete(base) and base.exists():
                state_runner = copy.copy(runner)
                state_runner.args = copy.copy(args)
                state_runner.args.latent_steps = 0
                try:
                    states[0] = prepare_state(
                        use_case.storage, state_runner, state, item, index, can_extend
                    )
                except StochasticForwardError:
                    save_json_atomic(invalid, {"reason": "stochastic_forward"})
                    raise
            for condition in conditions:
                condition_step = 0 if condition == "pe" else step
                ident = result_identity(
                    state.name, condition_step, condition, item, args, model
                )
                result_hash = canonical_hash(ident)
                directory = (
                    state
                    / "samples"
                    / sample
                    / "results"
                    / f"ls{condition_step}"
                    / result_hash
                    / condition
                )
                paths[condition].append(str(directory.relative_to(state)))
                if is_condition_complete(directory):
                    _, record = use_case.storage.load_intervention_result(
                        directory, ident
                    )
                else:
                    if condition_step not in states:
                        state_runner = copy.copy(runner)
                        state_runner.args = copy.copy(args)
                        state_runner.args.latent_steps = condition_step
                        try:
                            states[condition_step] = prepare_state(
                                use_case.storage,
                                state_runner,
                                state,
                                item,
                                index,
                                can_extend,
                                prepared,
                            )
                        except StochasticForwardError:
                            save_json_atomic(invalid, {"reason": "stochastic_forward"})
                            raise
                    kv, hidden, _ = states[condition_step]
                    seed = int(
                        canonical_hash({"identity": ident, "sample_index": index})[:16],
                        16,
                    )
                    device = torch.device(model.device)
                    devices = (
                        [
                            device.index
                            if device.index is not None
                            else torch.cuda.current_device()
                        ]
                        if device.type == "cuda"
                        else []
                    )
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        if devices:
                            torch.cuda.manual_seed_all(seed)
                        rng = {
                            "seed": seed,
                            "sample_index": index,
                            "condition": condition,
                        }
                        if condition in {"pe", "ple"}:
                            branch, record = runner.execute_ple(item, kv, hidden[-1])
                            record["condition"] = condition
                        else:
                            branch, record = getattr(runner, f"execute_{condition}")(
                                item, kv
                            )
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
        condition_step = 0 if condition == "pe" else step
        run_id = canonical_hash(
            {
                "state": state.name,
                "step": condition_step,
                "condition": condition,
                "results": paths[condition],
            }
        )
        run_dir = state / "runs" / run_id
        new_run = not (run_dir / "summary.json").is_file()
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "state_path": str(state.relative_to(use_case.storage.base_dir)),
            "step": condition_step,
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
                    state, f"{index:06d}", condition_step
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
