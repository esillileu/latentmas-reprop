"""Migrate MLflow names and path tags."""

from pathlib import Path

from .paths import new_relpath

EXPERIMENT_RENAMES = {
    "latentmas_intervention": "latentmas/intervention/kv",
    "latentmas_single_intervention": "latentmas/intervention/continuation",
    "latentmas_receiver_acquisition": "latentmas/representation/acquisition",
    "latentmas_receiver_trajectory": "latentmas/representation/trajectory",
    "latentmas_sender_probe": "latentmas/representation/sender_probe",
}


def _tag_updates(tags: dict[str, str], experiments_dir: Path) -> dict[str, str]:
    """New tag values for a run whose rel_dir is affected by the rename."""
    rel = tags.get("rel_dir")
    renamed = new_relpath(rel) if rel else None
    if not renamed:
        return {}
    updates = {"rel_dir": renamed}
    if tags.get("run_name") == rel:
        updates["run_name"] = renamed
    data_dir = tags.get("data_dir")
    if data_dir and data_dir.endswith(rel):
        updates["data_dir"] = data_dir[: -len(rel)] + renamed
    elif data_dir:
        updates["data_dir"] = str((experiments_dir / renamed).resolve())
    return updates


def run_mlflow(
    tracking_uri: str, experiments_dir: Path, execute: bool, rename_experiments: bool
) -> dict:
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    summary = {"runs_updated": 0, "experiments_renamed": [], "experiments_skipped": []}
    experiments = client.search_experiments(view_type=ViewType.ALL)
    for experiment in experiments:
        token = None
        while True:
            runs = client.search_runs(
                [experiment.experiment_id],
                run_view_type=ViewType.ALL,
                max_results=1000,
                page_token=token,
            )
            for run in runs:
                updates = _tag_updates(run.data.tags, experiments_dir)
                if not updates:
                    continue
                summary["runs_updated"] += 1
                print(f"[mlflow-run] {run.info.run_id} {updates}")
                if execute:
                    for key, value in updates.items():
                        client.set_tag(run.info.run_id, key, value)
                    if run.info.run_name == run.data.tags.get("rel_dir"):
                        client.update_run(run.info.run_id, name=updates["rel_dir"])
            token = runs.token
            if not token:
                break
    if rename_experiments:
        names = {e.name for e in experiments}
        for experiment in experiments:
            target = EXPERIMENT_RENAMES.get(experiment.name)
            if target is None:
                continue
            if target in names:
                summary["experiments_skipped"].append(
                    f"{experiment.name}: target {target} already exists"
                )
                print(f"[mlflow-experiment-skip] {experiment.name} -> {target} exists")
                continue
            print(f"[mlflow-experiment] {experiment.name} -> {target}")
            if execute:
                client.rename_experiment(experiment.experiment_id, target)
            summary["experiments_renamed"].append(f"{experiment.name} -> {target}")
    return summary
