"""Replace a legacy acquisition run with canonical sender-probe results."""

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import Metric, Param

from .analysis import ProbeAnalysisConfig, analyze_sender_states

LEGACY_PROBE_PARAMS = {
    "probe_epochs",
    "probe_prompt_templates",
    "probe_sender_latents",
    "probe_train_template_fraction",
}
PRESERVED_ARTIFACTS = (
    "results/sample_results.jsonl",
    "results/condition_aggregates.json",
    "tokenizer/candidate_token_mapping.json",
    "probe/sender_latent_states.pt",
)


def _trace_ids(client: MlflowClient, run_id: str, experiment_id: str) -> list[str]:
    result: list[str] = []
    token = None
    while True:
        page = client.search_traces(
            run_id=run_id,
            locations=[experiment_id],
            max_results=100,
            page_token=token,
            include_spans=False,
        )
        result.extend(trace.info.trace_id for trace in page)
        token = page.token
        if not token:
            return result


def _without_probe_results(value: Any) -> Any:
    if isinstance(value, dict):
        obsolete = {
            "chance_probe_accuracy",
            "probe_accuracy_by_step",
            "probe_metrics",
        }
        return {
            key: _without_probe_results(item)
            for key, item in value.items()
            if key not in obsolete
        }
    if isinstance(value, list):
        return [_without_probe_results(item) for item in value]
    return value


def _analysis_params(
    config: ProbeAnalysisConfig, analysis: dict[str, Any]
) -> dict[str, Any]:
    return {
        "probe_folds": config.folds,
        "probe_permutations": config.permutations,
        "probe_backend": analysis["resolved_backend"],
        "probe_solver": analysis["solver"],
        "probe_workers": analysis["resolved_workers"],
        "probe_c": config.c,
        "probe_max_iter": config.max_iter,
        "probe_tol": config.tol,
        "probe_batch_size": analysis["resolved_batch_size"],
    }


def _replacement_params(
    source_params: dict[str, str], settings: dict[str, Any]
) -> list[Param]:
    replaced = LEGACY_PROBE_PARAMS | settings.keys()
    params = [
        Param(key, value)
        for key, value in source_params.items()
        if key not in replaced
    ]
    params.extend(Param(key, str(value)) for key, value in settings.items())
    return params


def replace_probe_run(
    source_run_id: str,
    config: ProbeAnalysisConfig,
    *,
    client: MlflowClient | None = None,
) -> str:
    """Clone valid acquisition data, install canonical probe results, and retire source."""
    client = client or MlflowClient()
    source = client.get_run(source_run_id)
    experiment_id = source.info.experiment_id
    with tempfile.TemporaryDirectory(
        prefix=f"probe-replacement-{source_run_id[:8]}-"
    ) as temporary:
        states_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=source_run_id,
                artifact_path="probe/sender_latent_states.pt",
                dst_path=temporary,
            )
        )
        import torch

        payload = torch.load(states_path, map_location="cpu", weights_only=False)
        results, statistics = analyze_sender_states(payload, config)
        tags = {
            key: value
            for key, value in source.data.tags.items()
            if key not in {"mlflow.runName", "mlflow.runId"}
        }
        tags.update(
            {
                "probe_canonical": "true",
                "probe_schema": "grouped_oof_max_stat_v1",
                "replaces_run_id": source_run_id,
            }
        )
        replacement = client.create_run(
            experiment_id,
            start_time=source.info.start_time,
            tags=tags,
            run_name=source.data.tags.get("mlflow.runName"),
        )
        replacement_id = replacement.info.run_id
        try:
            _populate_replacement(
                client,
                source,
                replacement_id,
                temporary,
                config,
                results,
                statistics,
            )
            traces = _trace_ids(client, source_run_id, experiment_id)
            if traces:
                client.link_traces_to_run(traces, replacement_id)
            client.set_terminated(
                replacement_id, status="FINISHED", end_time=source.info.end_time
            )
            copied = client.get_run(replacement_id)
            stale_params = LEGACY_PROBE_PARAMS & copied.data.params.keys()
            if stale_params:
                raise RuntimeError(f"legacy probe parameters remain: {stale_params}")
            acquisition_metrics = {
                key for key in source.data.metrics if not key.startswith("probe/")
            }
            if not acquisition_metrics <= copied.data.metrics.keys():
                raise RuntimeError("replacement acquisition metrics are incomplete")
            probe_artifacts = {
                item.path for item in client.list_artifacts(replacement_id, "probe")
            }
            required_artifacts = {
                "probe/null_statistics.json",
                "probe/results.json",
                "probe/sender_latent_states.pt",
            }
            if not required_artifacts <= probe_artifacts:
                raise RuntimeError("replacement probe artifacts are incomplete")
            replacement_traces = _trace_ids(client, replacement_id, experiment_id)
            if set(replacement_traces) != set(traces):
                raise RuntimeError("replacement trace verification failed")
            client.delete_run(source_run_id)
            return replacement_id
        except Exception:
            client.set_terminated(replacement_id, status="FAILED")
            raise


def _populate_replacement(
    client: MlflowClient,
    source: Any,
    replacement_id: str,
    temporary: str,
    config: ProbeAnalysisConfig,
    results: dict[str, Any],
    statistics: dict[str, Any],
) -> None:
    analysis = results["analysis"]
    settings = _analysis_params(config, analysis)
    params = _replacement_params(source.data.params, settings)
    metrics = []
    for key in source.data.metrics:
        if not key.startswith("probe/"):
            metrics.extend(
                Metric(key, item.value, item.timestamp, item.step)
                for item in client.get_metric_history(source.info.run_id, key)
            )
    client.log_batch(replacement_id, metrics=metrics, params=params)
    for artifact_path in PRESERVED_ARTIFACTS:
        local = Path(
            mlflow.artifacts.download_artifacts(
                run_id=source.info.run_id,
                artifact_path=artifact_path,
                dst_path=temporary,
            )
        )
        client.log_artifact(replacement_id, local, str(Path(artifact_path).parent))
    summary_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=source.info.run_id,
            artifact_path="results/summary.json",
            dst_path=temporary,
        )
    )
    client.log_dict(
        replacement_id,
        _without_probe_results(json.loads(summary_path.read_text())),
        "results/summary.json",
    )
    config_path = Path(
        mlflow.artifacts.download_artifacts(
            run_id=source.info.run_id,
            artifact_path="config/resolved_config.yaml",
            dst_path=temporary,
        )
    )
    resolved = yaml.safe_load(config_path.read_text())
    for key in LEGACY_PROBE_PARAMS:
        resolved.pop(key, None)
    resolved.update(settings)
    client.log_dict(replacement_id, resolved, "config/resolved_config.yaml")
    client.log_dict(replacement_id, results, "probe/results.json")
    client.log_dict(replacement_id, statistics, "probe/null_statistics.json")
    for cell in results["cells"]:
        stem = f"probe/{cell['representation']}/step_{cell['step']}"
        for suffix, key in (
            ("accuracy", "observed_oof_accuracy"),
            ("raw_p_value", "raw_p_value"),
            ("fwer_p_value", "fwer_p_value"),
        ):
            client.log_metric(replacement_id, f"{stem}/{suffix}", cell[key])
