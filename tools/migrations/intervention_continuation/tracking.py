"""Read existing MLflow run references without modifying runs or traces."""

import contextlib
import sqlite3
from pathlib import Path
from urllib.parse import quote

from tools.migrations.use_case_names.migrate import new_relpath


def find_runs(uri: str, experiments_root: Path) -> dict[str, list[str]]:
    if uri.startswith("sqlite:///"):
        db = Path(uri[len("sqlite:///") :]).resolve()
        if not db.is_file():
            raise ValueError("MLflow database does not exist; refusing to create it")
        with contextlib.closing(
            sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True)
        ) as connection:
            rows = connection.execute(
                "SELECT r.run_uuid, t.key, t.value FROM runs r "
                "JOIN tags t ON t.run_uuid = r.run_uuid "
                "WHERE t.key IN ('data_dir', 'rel_dir')",
            )
            tags: dict[str, dict[str, str]] = {}
            for run_id, key, value in rows:
                tags.setdefault(run_id, {})[key] = value
        return _map_runs(tags, experiments_root)
    if not uri.startswith(("http://", "https://")):
        raise ValueError("Only HTTP(S) or read-only SQLite tracking URIs are supported")
    from mlflow.entities import ViewType
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=uri)
    tags = {}
    experiments = []
    token = None
    while True:
        page = client.search_experiments(
            view_type=ViewType.ALL, max_results=1000, page_token=token
        )
        experiments.extend(page)
        token = page.token
        if not token:
            break
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
                tags[run.info.run_id] = run.data.tags
            token = runs.token
            if not token:
                break
    return _map_runs(tags, experiments_root)


def _map_runs(
    tags: dict[str, dict[str, str]], experiments_root: Path
) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for run_id, values in tags.items():
        candidates = {values.get("rel_dir", "")}
        data_dir = values.get("data_dir")
        if data_dir:
            with contextlib.suppress(ValueError):
                candidates.add(str(Path(data_dir).relative_to(experiments_root)))
        for candidate in candidates:
            if candidate:
                renamed = new_relpath(candidate) or candidate
                mapping.setdefault(renamed, []).append(run_id)
    return {key: sorted(set(value)) for key, value in mapping.items()}
