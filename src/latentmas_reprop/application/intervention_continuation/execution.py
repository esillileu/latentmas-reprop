"""Persist and reuse intervention continuation states and condition results."""

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
from tqdm import tqdm

from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .budget_reuse import get_or_create_result, should_compute_condition
from .checkpoint import (
    _check_sample,
    _sample_identity,
    prepare_sample_states,
    prepare_state,
)
from .determinism import StochasticForwardError
from .identity import (
    canonical_hash,
    condition_name,
    ensure_state,
    resolve_conditions,
    result_identity,
    sample_seed,
    state_identity,
    state_path,
)

_prepare_sample = prepare_sample_states


def execute(
    use_case: Any, model: Any, args: Any
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from .artifacts import (
        build_summary_dict,
        save_experiment_summary,
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
    conditions = resolve_conditions(args)
    step = int(getattr(args, "latent_steps", 10))
    runner = InterventionContinuationRunner(model, args, use_case.evaluator_port)
    records = []
    paths = {condition: [] for condition in conditions}
    tracking_ctx = None
    if use_case.tracker_port:
        from .tracking import start_continuation_staging_tracker

        tracking_ctx = start_continuation_staging_tracker(
            use_case.tracker_port, use_case.storage.base_dir, state, args, model
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = None
        for index, item in enumerate(
            tqdm(
                dataset,
                desc="Intervention continuation",
                unit="sample",
                disable=bool(getattr(args, "no_progress", False)),
            )
        ):
            prepared = future.result() if future is not None else {}
            future = None
            if index + 1 < len(dataset):
                next_index = index + 1
                next_item = dataset[next_index]
                next_sample = f"{next_index:06d}"
                needed = set()
                for condition in conditions:
                    condition_step = 0 if condition in {"qe", "rqe", "rqrqe"} else step
                    ident = result_identity(
                        state.name, condition_step, condition, next_item, args, model
                    )
                    if should_compute_condition(
                        state, next_sample, condition_step, condition, ident
                    ):
                        needed.add(condition_step)
                if needed:
                    future = executor.submit(
                        prepare_sample_states,
                        use_case.storage,
                        state,
                        next_sample,
                        needed,
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
                condition_step = 0 if condition in {"qe", "rqe", "rqrqe"} else step
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
                record = get_or_create_result(
                    use_case, directory, state, sample, condition_step, ident
                )
                if record is None:
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
                    seed = sample_seed(
                        int(getattr(args, "seed", 42)),
                        identity["task"],
                        identity["split"],
                        index,
                        item["question"],
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
                        torch.random.default_generator.manual_seed(seed)
                        if devices:
                            with torch.cuda.device(device):
                                torch.cuda.manual_seed(seed)
                        rng = {
                            "seed": seed,
                            "policy": "sample_v2",
                            "sample_index": index,
                            "condition": condition,
                        }
                        if condition in {"qe", "rqe", "qle", "rqle"}:
                            branch, record = runner.execute_unprompted(
                                item, kv, hidden[-1]
                            )
                            record["condition"] = condition
                        elif condition in {"rqlrqe", "rqlre", "rqlqe", "rqrqe"}:
                            branch, record = runner.execute_prompt(item, kv, condition)
                        else:
                            branch, record = runner.execute_control(item, kv)
                            record["condition"] = condition
                    from ...domain.services.prompts import initial_role

                    base_meta = load_json(base / "result.json")
                    record = dict(
                        record,
                        condition_name=condition_name(condition, args),
                        initial_role=initial_role(args),
                        initial_prompt=base_meta.get("prompt", ""),
                        initial_prompt_token_ids=base_meta.get("prompt_token_ids", []),
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
            if tracking_ctx is not None and use_case.tracker_port:
                from .tracking import trace_continuation_sample

                try:
                    trace_continuation_sample(
                        tracker=use_case.tracker_port,
                        state=state,
                        sample=sample,
                        index=index,
                        item=item,
                        args=args,
                        model=model,
                        group=tracking_ctx["group"],
                        config=tracking_ctx["config"],
                        config_path=tracking_ctx["config_path"],
                    )
                except Exception:
                    use_case.tracker_port.discard_staging_run(status="FAILED")
                    raise
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        condition_step = 0 if condition in {"qe", "rqe", "rqrqe"} else step
        run_id = canonical_hash(
            {
                "state": state.name,
                "step": condition_step,
                "condition": condition,
                "results": paths[condition],
            }
        )
        run_dir = state / "runs" / run_id
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
    summary = build_summary_dict(records, conditions, len(dataset), vars(args))
    if tracking_ctx is not None and use_case.tracker_port:
        try:
            from .tracking import promote_continuation_tracker

            promote_continuation_tracker(
                tracker=use_case.tracker_port,
                tracking_ctx=tracking_ctx,
                metrics=summary["metrics"],
                args=args,
                model=model,
                state=state,
            )
        except Exception:
            use_case.tracker_port.discard_staging_run(status="FAILED")
            raise
    return summary, records
