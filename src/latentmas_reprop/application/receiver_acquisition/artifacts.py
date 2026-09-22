"""Artifact logging and MLflow tracking for receiver acquisition experiments."""

import json
from typing import Any

from ...domain.models import ReceiverAcquisitionMetrics, ReceiverAcquisitionRecord
from ...domain.ports.cache_port import CacheLayer, CachePort
from ...domain.ports.tracking_port import ExperimentTrackerPort
from ..sender_latent_probe import SenderProbeResult


def log_receiver_artifacts(
    cache_port: CachePort,
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    records: list[ReceiverAcquisitionRecord],
    metrics: ReceiverAcquisitionMetrics,
    mapping: dict[str, Any],
    hidden: dict[str, tuple],
    probe_result: SenderProbeResult | None,
) -> None:
    """Save acquisition json results, summary, configurations, and upload artifacts."""
    data = [record.to_dict() for record in records]
    cache_port.save_json(
        CacheLayer.EVALUATION_RECEIVER_ACQUISITION, "sample_results.json", data
    )
    cache_port.save_json(
        CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
        "summary.json",
        metrics.to_dict(),
    )
    if tracker_port:
        tracker_port.log_dict(metrics.to_dict(), "results/summary.json")
        tracker_port.log_dict(
            {"cells": metrics.cells}, "results/condition_aggregates.json"
        )
        tracker_port.log_dict(mapping, "tokenizer/candidate_token_mapping.json")
        tracker_port.log_dict(vars(args), "config/resolved_config.yaml")
        path = (
            cache_port.get_layer_path(CacheLayer.EVALUATION_RECEIVER_ACQUISITION)
            / "sample_results.jsonl"
        )
        path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n",
            encoding="utf-8",
        )
        tracker_port.log_artifact(path, "results")
        if hidden:
            hidden_path = cache_port.save_torch(
                CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
                "receiver_hidden_states.pt",
                hidden,
            )
            tracker_port.log_artifact(hidden_path, "states")

    if probe_result is not None:
        log_probe_artifacts(cache_port, tracker_port, args, probe_result)


def log_probe_artifacts(
    cache_port: CachePort,
    tracker_port: ExperimentTrackerPort | None,
    args: Any,
    probe_result: SenderProbeResult,
) -> None:
    """Log linear probe artifacts, splits, and confusion matrices."""
    state_payload = {
        **probe_result.states,
        "metadata": probe_result.metadata,
    }
    if args.save_latent_states:
        state_path = cache_port.save_torch(
            CacheLayer.EVALUATION_RECEIVER_ACQUISITION,
            "sender_latent_states.pt",
            state_payload,
        )
        if tracker_port:
            tracker_port.log_artifact(state_path, "probe")
    if tracker_port:
        tracker_port.log_param("analysis_stage", "carrier_probe")
        tracker_port.log_dict(probe_result.split, "probe/probe_split.json")
        tracker_port.log_dict(probe_result.summary, "probe/probe_summary.json")
        for name, matrix in probe_result.confusion_matrices.items():
            tracker_port.log_dict(
                {"labels": list(range(10)), "matrix": matrix},
                f"probe/confusion_matrix_{name}.json",
            )
