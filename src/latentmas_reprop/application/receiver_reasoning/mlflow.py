"""Select a saved MLflow run and download its analysis inputs."""

import json
from pathlib import Path

import yaml


def select_runs(client, *, version_tag, experiment_name="latentmas_receiver_reasoning"):
    if not version_tag or not version_tag.strip():
        raise ValueError("--version-tag is required and must be nonempty")
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    selected, page_token = [], None
    while True:
        page = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            max_results=1000,
            page_token=page_token,
        )
        selected.extend(
            run.info.run_id
            for run in page
            if run.data.tags.get("version_tag") == version_tag
        )
        page_token = getattr(page, "token", None)
        if not page_token:
            break
    if not selected:
        raise ValueError(f"No FINISHED MLflow runs with version_tag={version_tag!r}")
    return selected


def load_run(client, paths, *, run_id):
    run = client.get_run(run_id)
    artifacts = {a.path for a in client.list_artifacts(run_id, "results")}
    required = {"results/sample_results.jsonl", "results/resolved_config.yaml"}
    if not required <= artifacts:
        raise ValueError(
            f"Run {run_id} missing artifacts: {sorted(required - artifacts)}"
        )
    cache = paths.get_cache_layer_dir(f"receiver_reasoning/mlflow/{run_id}")
    # Refresh from MLflow so a stale local file cannot substitute for the source.
    downloaded = {
        name: Path(client.download_artifacts(run_id, name, dst_path=str(cache)))
        for name in sorted(required)
    }
    records = [
        json.loads(line)
        for line in downloaded["results/sample_results.jsonl"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    config = yaml.safe_load(
        downloaded["results/resolved_config.yaml"].read_text(encoding="utf-8")
    )
    metadata = {
        "run_id": run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "artifact_uri": run.info.artifact_uri,
        "params": dict(run.data.params),
        "metrics": dict(run.data.metrics),
        "tags": dict(run.data.tags),
    }
    return records, config, metadata
