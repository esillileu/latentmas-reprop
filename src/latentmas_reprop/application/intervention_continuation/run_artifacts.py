"""Publish per-condition intervention run artifacts."""

from ...infrastructure.storage.tensor_store import load_json, save_json_atomic
from .artifacts import build_summary_dict, save_experiment_summary
from .identity import canonical_hash


def save_condition_runs(
    state, base_dir, conditions, step, paths, records, sample_count
):
    for condition in conditions:
        subset = [record for record in records if record["condition"] == condition]
        condition_step = subset[0]["identity"]["step"] if subset else step
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
            "state_path": str(state.relative_to(base_dir)),
            "step": condition_step,
            "condition": condition,
            "result_paths": paths[condition],
        }
        config_path = run_dir / "config.json"
        if config_path.is_file() and load_json(config_path) != config:
            raise ValueError("Run identity mismatch")
        if not config_path.is_file():
            save_json_atomic(config_path, config)
        summary = build_summary_dict(subset, [condition], sample_count, config)
        save_experiment_summary(run_dir, summary, subset)
