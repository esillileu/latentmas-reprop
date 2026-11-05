"""Compare single-agent and Sequential MAS continuations using their own states."""

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .artifacts import build_summary_dict, save_experiment_summary
from .conditions import COMPARISON_CONDITIONS, branch_step, comparison_groups
from .identity import (
    canonical_hash,
    ensure_state,
    resolve_conditions,
    state_identity,
    state_path,
)


def execute_comparison(use_case: Any, model: Any, args: Any) -> tuple[dict, list[dict]]:
    """Execute existing state orchestration twice and publish one comparison run."""
    from .execution import execute
    from .tracking import _group, track_results

    if getattr(args, "overwrite", False):
        raise ValueError("Overwriting shared states is not supported")
    conditions = resolve_conditions(args)
    base = use_case.storage.base_dir
    identities, state_paths = {}, {}
    for group in comparison_groups(args, list(COMPARISON_CONDITIONS)):
        ident, provenance = state_identity(model, group, "transformers")
        identities[group.initial_role] = ident
        state_paths[group.initial_role] = str(state_path(base, ident).relative_to(base))
    identity = {
        "model": args.model_name,
        "task": getattr(args, "task", "gsm8k"),
        "split": getattr(args, "split", "test"),
        "initial_states": identities,
    }
    state = state_path(base, identity)
    ensure_state(state, identity, provenance)
    dataset = list(
        use_case.dataset_port.load(task=identity["task"], split=identity["split"])
    )
    limit = int(getattr(args, "max_samples", -1))
    if limit > 0:
        dataset = dataset[:limit]
    child = copy.copy(use_case)
    child.tracker_port = None
    child.dataset_port = SimpleNamespace(load=lambda **kwargs: dataset)
    records = []
    for group in comparison_groups(args, conditions):
        _, completed = execute(child, model, group)
        records.extend(completed)
    records.sort(
        key=lambda record: (
            record["sample_index"],
            conditions.index(record["condition"]),
        )
    )
    group_hash = canonical_hash(_group(state, args, model))
    run_dir = state / "runs" / group_hash
    config_path = run_dir / "config.json"
    config = (
        load_json(config_path)
        if config_path.is_file()
        else {
            "state_path": str(state.relative_to(base)),
            "state_paths": state_paths,
            "step": int(getattr(args, "latent_steps", 60)),
            "condition_steps": {
                condition: branch_step(condition, args)
                for condition in COMPARISON_CONDITIONS
            },
            "result_paths": [],
            "result_references": {},
        }
    )
    for record in records:
        ident = record["identity"]
        source = Path(state_paths[record["initial_role"]])
        path = (
            source
            / "samples"
            / record["sample_id"]
            / "results"
            / f"ls{ident['step']}"
            / canonical_hash(ident)
            / record["condition"]
        )
        sample = record["sample_id"]
        references = set(config["result_references"].get(sample, []))
        references.add(path.as_posix())
        config["result_references"][sample] = sorted(references)
    save_json_atomic(config_path, config)
    # Include previously completed conditions when extending the same comparison.
    all_records = []
    for index, item in enumerate(dataset):
        all_records.extend(
            comparison_records(base, config, f"{index:06d}", item).values()
        )
    completed_conditions = [
        condition
        for condition in COMPARISON_CONDITIONS
        if any(record["condition"] == condition for record in all_records)
    ]
    summary = build_summary_dict(
        all_records, completed_conditions, len(dataset), config
    )
    save_experiment_summary(run_dir, summary, all_records)
    if use_case.tracker_port:
        track_results(
            use_case.tracker_port, base, state, dataset, args, model, summary["metrics"]
        )
    return summary, records


def comparison_records(
    base: Path, config: dict, sample: str, item: dict
) -> dict[str, dict]:
    """Read completed references without copying tensor artifacts."""
    records = {}
    for relative in config["result_references"].get(sample, []):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Invalid comparison result reference")
        if len(path.parts) < 6 or path.parts[-6:-3] != ("samples", sample, "results"):
            raise ValueError("Comparison reference belongs to another sample")
        directory = base / path
        if not is_condition_complete(directory):
            raise ValueError(f"Incomplete comparison result: {directory}")
        record = load_json(directory / "result.json")
        identity = record["identity"]
        if (
            record.get("status") != "complete"
            or record.get("sample_index") != int(sample)
            or identity["gold"] != item.get("gold", "")
            or path.parts[-2] != canonical_hash(identity)
        ):
            raise ValueError(
                f"Comparison sample or result identity mismatch: {directory}"
            )
        shared = load_json(directory.parents[3] / "shared" / "ls0" / "result.json")
        if shared["question"] != item["question"]:
            raise ValueError("Comparison question mismatch")
        records[record["condition_name"]] = record
    return records


def comparison_meta(base: Path, config: dict, sample: str, records: dict) -> dict:
    """Expose both actual initial states instead of claiming one common prefill."""
    states = {}
    for role in {record["initial_role"] for record in records.values()}:
        path = (
            base / config["state_paths"][role] / "samples" / sample / "shared" / "ls0"
        )
        states[role] = load_json(path / "result.json")
    return {"initial_input": "comparison", "initial_states": states}
