"""Artifact logging and MLflow tracking for receiver acquisition experiments."""

import contextlib
import json
from typing import Any

import torch

from ...domain.models import ReceiverAcquisitionMetrics, ReceiverAcquisitionRecord
from ...domain.ports.cache_port import CachePort
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ...infrastructure.paths.naming import build_experiment_relpath
from ...infrastructure.paths.resolver import get_path_resolver


def log_receiver_artifacts(
    cache_port: CachePort,
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    records: list[ReceiverAcquisitionRecord],
    metrics: ReceiverAcquisitionMetrics,
    mapping: dict[str, Any],
    hidden: dict[str, tuple],
    sender_states: dict[str, Any] | None,
) -> Any:
    """Save acquisition json results, summary, configurations, and upload artifacts."""
    data = [record.to_dict() for record in records]

    exp_relpath = getattr(args, "exp_relpath", None) or build_experiment_relpath(
        args, backend="transformers"
    )
    exp_dir = get_path_resolver().get_experiment_dir(exp_relpath)
    exp_dir.mkdir(parents=True, exist_ok=True)

    sample_results_json = exp_dir / "sample_results.json"
    sample_results_json.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_json = exp_dir / "summary.json"
    summary_json.write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    sample_results_jsonl = exp_dir / "sample_results.jsonl"
    sample_results_jsonl.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n",
        encoding="utf-8",
    )

    if hidden:
        hidden_path = exp_dir / "receiver_hidden_states.pt"
        torch.save(hidden, hidden_path)
        if tracker_port:
            tracker_port.log_artifact(hidden_path, "states")

    state_path = None
    if sender_states is not None and getattr(args, "save_latent_states", False):
        probe_dir = exp_dir / "probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        state_path = probe_dir / "sender_latent_states.pt"
        torch.save(sender_states, state_path)

        with contextlib.suppress(Exception):
            if not (exp_dir / "sender_latent_states.pt").is_file():
                torch.save(sender_states, exp_dir / "sender_latent_states.pt")

        if tracker_port:
            tracker_port.log_artifact(state_path, "probe")

    if tracker_port:
        tracker_port.log_dict(metrics.to_dict(), "results/summary.json")
        tracker_port.log_dict(
            {"cells": metrics.cells}, "results/condition_aggregates.json"
        )
        tracker_port.log_dict(mapping, "tokenizer/candidate_token_mapping.json")
        tracker_port.log_dict(vars(args), "config/resolved_config.yaml")
        tracker_port.log_artifact(sample_results_jsonl, "results")

    return (
        state_path
        if sender_states is not None and getattr(args, "save_latent_states", False)
        else None
    )
