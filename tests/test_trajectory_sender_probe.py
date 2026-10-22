"""Bind saved sender-probe analysis to the trajectory experiment and source run."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmas_reprop.application.sender_probe import ProbeAnalysisConfig, use_case
from src.run import sender_probe as cli


def setup_cli(monkeypatch, tmp_path, status="FINISHED", phase="sweep"):
    source = SimpleNamespace(
        info=SimpleNamespace(experiment_id="trajectory", status=status),
        data=SimpleNamespace(tags={"phase": phase}, params={}),
    )
    client = SimpleNamespace(
        get_run=lambda run_id: source,
        get_experiment=lambda experiment_id: SimpleNamespace(
            name="latentmas_receiver_trajectory"
        ),
    )
    monkeypatch.setattr(cli.mlflow, "MlflowClient", lambda: client)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example")
    monkeypatch.setattr(cli, "MLflowTracker", lambda **kwargs: object())
    downloads, calls = [], []

    def download(**kwargs):
        assert kwargs["run_id"] == "sweep-14B"
        downloads.append(kwargs["artifact_path"])
        path = tmp_path / kwargs["artifact_path"]
        if path.name == "source.json":
            path.write_text(
                json.dumps(
                    {"model": "Qwen/Qwen3-14B", "smoke": False, "sample_count": 100}
                )
            )
        else:
            path.touch()
        return str(path)

    monkeypatch.setattr(cli.mlflow.artifacts, "download_artifacts", download)

    class Analysis:
        def __init__(self, tracker):
            pass

        def execute(self, *args, **kwargs):
            calls.append(kwargs)
            return {}

    monkeypatch.setattr(cli, "SenderProbeAnalysisUseCase", Analysis)
    return downloads, calls


def test_cli_uses_sweep_root_states_and_same_experiment(monkeypatch, tmp_path):
    downloads, calls = setup_cli(monkeypatch, tmp_path)
    assert cli.main(["--source-run-id", "sweep-14B"]) == 0
    assert downloads == ["source.json", "sender_latent_states.pt"]
    assert calls[0]["source_run_id"] == "sweep-14B"
    assert calls[0]["source_artifact_path"] == "sender_latent_states.pt"
    assert calls[0]["experiment_name"] == "latentmas_receiver_trajectory"
    assert calls[0]["source_config"]["model"] == "Qwen/Qwen3-14B"


@pytest.mark.parametrize(
    "extra",
    (
        ["--source-artifact-path", "probe/sender_latent_states.pt"],
        ["--source-config", "different.yaml"],
        ["--replace-source-run"],
    ),
)
def test_cli_rejects_foreign_input_and_replacement(monkeypatch, tmp_path, extra):
    downloads, calls = setup_cli(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "_read_config", lambda path: {})
    with pytest.raises(ValueError):
        cli.main(["--source-run-id", "sweep-14B", *extra])
    assert not downloads and not calls


@pytest.mark.parametrize(
    "status,phase", (("RUNNING", "sweep"), ("FINISHED", "sender_probe"))
)
def test_cli_requires_completed_collection(monkeypatch, tmp_path, status, phase):
    downloads, calls = setup_cli(monkeypatch, tmp_path, status, phase)
    with pytest.raises(ValueError):
        cli.main(["--source-run-id", "sweep-14B"])
    assert not downloads and not calls


def test_use_case_logs_sweep_lineage_in_trajectory_experiment(monkeypatch, tmp_path):
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "hidden_pre_realign": torch.zeros(200, 20, 1),
            "latent_post_realign": torch.zeros(200, 20, 1),
        },
    )
    results = {
        "analysis": {
            "resolved_backend": "sklearn",
            "solver": "test",
            "resolved_workers": 1,
        },
        "cells": [
            {
                "representation": "latent_post_realign",
                "step": k,
                "observed_oof_accuracy": 0.1,
            }
            for k in range(1, 21)
        ],
    }
    monkeypatch.setattr(
        use_case, "analyze_sender_states", lambda *args: (results, {"saved": True})
    )
    starts, params, artifacts, statuses = [], [], [], []
    tracker = SimpleNamespace(
        start_run=lambda **kwargs: starts.append(kwargs),
        log_params=lambda values: params.append(values),
        log_dict=lambda value, path: artifacts.append(path),
        log_metrics=lambda values: None,
        end_run=lambda status: statuses.append(status),
    )
    use_case.SenderProbeAnalysisUseCase(tracker).execute(
        Path("unused.pt"),
        ProbeAnalysisConfig(),
        source_run_id="sweep",
        source_artifact_path="sender_latent_states.pt",
        source_config={"model": "Qwen/Qwen3-14B"},
        experiment_name="latentmas_receiver_trajectory",
    )
    assert starts[0]["experiment_name"] == "latentmas_receiver_trajectory"
    assert starts[0]["tags"]["phase"] == "sender_probe"
    assert starts[0]["tags"]["source_run_id"] == "sweep"
    assert params[0]["source_artifact"] == "sender_latent_states.pt"
    assert params[0]["model"] == "Qwen/Qwen3-14B"
    assert artifacts == [
        "probe/results.json",
        "probe/null_statistics.json",
        "source/resolved_config.yaml",
    ]
    assert statuses == ["FINISHED"]
