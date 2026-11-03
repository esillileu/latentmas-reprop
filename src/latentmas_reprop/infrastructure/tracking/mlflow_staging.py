import os
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import mlflow


def truncate_preview(text: str | None, max_length: int = 1000) -> str | None:
    """Truncate preview string if exceeding maximum allowed characters."""
    if text is None or len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def resolve_tracking_setup(
    resolver: Any,
    tracking_uri: str | None,
    artifact_location: str | None,
) -> tuple[str, bool, str | None]:
    """Resolve tracking URI, remote flag, and artifact location."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri is not None:
        resolved_uri = tracking_uri
    elif env_uri is not None and env_uri.strip():
        resolved_uri = env_uri.strip()
    else:
        db_path = resolver.mlflow_db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_uri = f"sqlite:///{db_path.resolve()}"

    is_remote = not resolved_uri.startswith(("sqlite", "file"))
    resolved_loc = None
    if not is_remote:
        resolved_loc = artifact_location or str(resolver.mlflow_artifacts_dir.resolve())
        Path(resolved_loc).mkdir(parents=True, exist_ok=True)
    return resolved_uri, is_remote, resolved_loc


def resolve_experiment_id(
    experiment_name: str,
    artifact_location: str | None,
    is_remote: bool,
) -> str:
    """Get or create an experiment and return its experiment ID."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if (
        experiment is not None
        and getattr(experiment, "lifecycle_stage", "") == "deleted"
    ):
        with suppress(Exception):
            mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
        experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        loc = artifact_location if not is_remote else None
        return (
            mlflow.create_experiment(experiment_name, artifact_location=loc)
            if loc
            else mlflow.create_experiment(experiment_name)
        )
    return experiment.experiment_id


def find_existing_run_id(
    client: Any,
    experiment_id: str,
    rel_dir: str | None,
    data_dir: str | None,
    run_name: str | None,
    is_staging: bool = False,
) -> str | None:
    """Find an existing run ID matching relative directory, data directory, or run name."""
    filters = [f"tags.rel_dir = '{rel_dir}'"] if rel_dir else []
    if data_dir:
        filters.append(f"tags.data_dir = '{data_dir}'")
    if run_name:
        filters.extend(
            [f"tags.run_name = '{run_name}'", f"attributes.run_name = '{run_name}'"]
        )

    for filt in filters:
        with suppress(Exception):
            runs = client.search_runs(
                experiment_ids=[experiment_id], filter_string=filt, max_results=10
            )
            for r in runs:
                r_is_staging = r.data.tags.get("is_staging") == "true"
                if is_staging == r_is_staging:
                    return r.info.run_id
    return None


def safe_log_params(client: Any, active_run: Any, params: dict[str, Any]) -> None:
    """Safely log parameters, updating changed params as tags to avoid MLflow conflicts."""
    safe = {
        k: (
            "None"
            if v is None
            else str(v)[:500]
            if isinstance(v, (list, tuple, dict)) or len(str(v)) > 500
            else v
        )
        for k, v in params.items()
    }
    active = active_run or mlflow.active_run()
    if active is not None:
        with suppress(Exception):
            run = client.get_run(active.info.run_id)
            existing = run.data.params
            new_params = {k: v for k, v in safe.items() if k not in existing}
            changed = {
                k: str(v)
                for k, v in safe.items()
                if k in existing and str(existing[k]) != str(v)
            }
            if changed:
                mlflow.set_tags(changed)
            safe = new_params
    if safe:
        with suppress(Exception):
            mlflow.log_params(safe)


def search_all_run_traces(
    client: Any, run_id: str, experiment_id: str | None = None
) -> list[str]:
    """Retrieve all trace IDs associated with a specific run."""
    collected: list[str] = []
    token = None
    if not experiment_id:
        with suppress(Exception):
            run = client.get_run(run_id)
            experiment_id = run.info.experiment_id
    if not experiment_id:
        return collected

    while True:
        try:
            page = client.search_traces(
                locations=[experiment_id],
                run_id=run_id,
                max_results=100,
                page_token=token,
                include_spans=False,
            )
        except Exception:
            break
        collected.extend([t.info.trace_id for t in page])
        token = getattr(page, "token", None)
        if not token:
            break
    return collected


def link_traces_in_chunks(
    client: Any, trace_ids: Sequence[str], target_run_id: str
) -> None:
    """Link traces to target run in chunks of 100."""
    for i in range(0, len(trace_ids), 100):
        chunk = list(trace_ids[i : i + 100])
        with suppress(Exception):
            client.link_traces_to_run(trace_ids=chunk, run_id=target_run_id)


def execute_staging_promotion(
    tracker: Any,
    staging_run_id: str | None,
    experiment_name: str | None,
    final_run_name: str,
    tags: dict[str, Any] | None = None,
    metrics: dict[str, float | int] | None = None,
    params: dict[str, Any] | None = None,
    artifacts: list[tuple[Path | str, str | None]] | None = None,
    status: str = "FINISHED",
) -> Any:
    """Promote collected staging traces into a finalized canonical run."""
    tracker.flush_traces()
    collected_trace_ids: list[str] = []
    if staging_run_id:
        with suppress(Exception):
            collected_trace_ids = search_all_run_traces(tracker.client, staging_run_id)

    tracker.end_run("FINISHED")

    final_tags = dict(tags or {})
    final_tags.pop("is_staging", None)
    if staging_run_id:
        final_tags["promoted_from"] = str(staging_run_id)

    final_run = tracker.start_run(
        experiment_name=experiment_name or "latentmas-reprop",
        run_name=final_run_name,
        tags=final_tags,
        is_staging=False,
    )
    final_run_id = tracker.active_run_id

    if collected_trace_ids and final_run_id:
        link_traces_in_chunks(tracker.client, collected_trace_ids, final_run_id)

    if params:
        tracker.log_params(params)
    if metrics:
        tracker.log_metrics(metrics)
    if artifacts:
        for local_path, art_path in artifacts:
            tracker.log_artifact(local_path, artifact_path=art_path)

    tracker.end_run(status=status)

    if staging_run_id and staging_run_id != final_run_id:
        with suppress(Exception):
            tracker.client.delete_run(staging_run_id)

    return final_run
