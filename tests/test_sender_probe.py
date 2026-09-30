"""Tests for grouped sender-state probe analysis."""

from types import SimpleNamespace

import numpy as np
import torch

import latentmas_reprop.application.sender_probe.replacement as replacement_module
from latentmas_reprop.application.sender_probe import (
    ProbeAnalysisConfig,
    analyze_sender_states,
)
from latentmas_reprop.application.sender_probe.folds import grouped_folds
from latentmas_reprop.application.sender_probe.replacement import (
    LEGACY_PROBE_PARAMS,
    _analysis_params,
    _replacement_params,
    _without_probe_results,
)
from latentmas_reprop.application.sender_probe.torch_solver import (
    TorchSolverConfig,
    _fit_predict,
)


def test_analysis_cli_does_not_import_model_wrapper():
    import subprocess
    import sys

    command = (
        "import sys; import run.sender_probe; "
        "assert 'latentmas_reprop.infrastructure.models.model_wrapper' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)


def _payload(signal: bool, seed: int = 4) -> dict:
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(10), 5)
    templates = np.repeat(np.arange(5), 10)
    states = rng.normal(0, 0.2, (50, 1, 10))
    if signal:
        states[:, 0, :] += np.eye(10)[labels] * 4
    tensor = torch.tensor(states, dtype=torch.float32)
    return {
        "labels": torch.tensor(labels),
        "template_indices": torch.tensor(templates),
        "hidden_pre_realign": tensor,
        "latent_post_realign": tensor.clone(),
        "metadata": [],
    }


def test_grouped_folds_are_disjoint_and_cover_oof_once():
    templates = np.repeat(np.arange(10), 10)
    folds = grouped_folds(templates, seed=42)
    tested = []
    for fold in folds:
        assert set(fold["train_template_indices"]).isdisjoint(
            fold["test_template_indices"]
        )
        tested.extend(fold["test_indices"])
    assert sorted(tested) == list(range(100))


def test_replacement_removes_legacy_probe_summary_fields():
    cleaned = _without_probe_results(
        {
            "runtime_total_sec": 3.0,
            "probe_metrics": {"legacy": 0.5},
            "research_matrix": {
                "probe_accuracy_by_step": {"legacy": 0.5},
                "carrier_probability_deltas": {"full": 0.2},
            },
        }
    )
    assert cleaned == {
        "runtime_total_sec": 3.0,
        "research_matrix": {"carrier_probability_deltas": {"full": 0.2}},
    }


def test_replacement_does_not_restore_legacy_probe_parameters():
    params = _analysis_params(
        ProbeAnalysisConfig(),
        {
            "resolved_backend": "sklearn",
            "solver": "lbfgs",
            "resolved_workers": 1,
            "resolved_batch_size": 256,
        },
    )
    assert params.keys().isdisjoint(LEGACY_PROBE_PARAMS)


def test_replacement_overwrites_analysis_parameters_without_duplicates():
    params = _replacement_params(
        {
            "model_name": "Qwen/Qwen3-0.6B",
            "probe_sender_latents": "True",
            "probe_backend": "auto",
            "probe_permutations": "5000",
        },
        {"probe_backend": "sklearn", "probe_permutations": 1},
    )
    values = {param.key: param.value for param in params}

    assert len(params) == len(values)
    assert values == {
        "model_name": "Qwen/Qwen3-0.6B",
        "probe_backend": "sklearn",
        "probe_permutations": "1",
    }


def test_replacement_soft_deletes_source_without_unlinking_traces(monkeypatch):
    source = SimpleNamespace(
        info=SimpleNamespace(
            run_id="source", experiment_id="experiment", start_time=1, end_time=2
        ),
        data=SimpleNamespace(tags={}, params={}, metrics={}),
    )
    copied = SimpleNamespace(data=SimpleNamespace(params={}, metrics={}))

    class Client:
        deleted_run_id = None

        def get_run(self, run_id):
            return source if run_id == "source" else copied

        def create_run(self, *args, **kwargs):
            return SimpleNamespace(info=SimpleNamespace(run_id="replacement"))

        def link_traces_to_run(self, trace_ids, run_id):
            assert trace_ids == ["trace"]
            assert run_id == "replacement"

        def set_terminated(self, *args, **kwargs):
            pass

        def list_artifacts(self, run_id, path):
            return [
                SimpleNamespace(path="probe/null_statistics.json"),
                SimpleNamespace(path="probe/results.json"),
                SimpleNamespace(path="probe/sender_latent_states.pt"),
            ]

        def delete_run(self, run_id):
            self.deleted_run_id = run_id

    client = Client()
    monkeypatch.setattr(
        replacement_module.mlflow.artifacts,
        "download_artifacts",
        lambda **kwargs: "/tmp/sender_latent_states.pt",
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        replacement_module, "analyze_sender_states", lambda *args: ({}, {})
    )
    monkeypatch.setattr(replacement_module, "_populate_replacement", lambda *args: None)
    monkeypatch.setattr(replacement_module, "_trace_ids", lambda *args: ["trace"])

    replacement_id = replacement_module.replace_probe_run(
        "source", ProbeAnalysisConfig(), client=client
    )

    assert replacement_id == "replacement"
    assert client.deleted_run_id == "source"


def test_batched_solver_matches_independent_fits():
    generator = torch.Generator().manual_seed(3)
    train_x = torch.randn(40, 6, generator=generator)
    test_x = torch.randn(10, 6, generator=generator)
    labels = torch.stack((torch.arange(40) % 10, torch.arange(39, -1, -1) % 10))
    config = TorchSolverConfig(
        c=1.0, max_iter=100, tol=1e-4, batch_size=2, device="cpu"
    )
    batched = _fit_predict(train_x, labels, test_x, config)
    separate = torch.cat(
        [_fit_predict(train_x, row.unsqueeze(0), test_x, config) for row in labels]
    )
    assert torch.equal(batched, separate)


def test_signal_is_max_stat_significant_and_reproducible():
    config = ProbeAnalysisConfig(seed=7, permutations=24, backend="sklearn", workers=1)
    first, first_null = analyze_sender_states(_payload(True), config)
    second, second_null = analyze_sender_states(_payload(True), config)
    assert first == second
    assert first_null == second_null
    assert all(cell["observed_oof_accuracy"] > 0.9 for cell in first["cells"])
    assert all(cell["fwer_p_value"] < 0.05 for cell in first["cells"])


def test_independent_states_are_not_corrected_significant():
    result, _ = analyze_sender_states(
        _payload(False),
        ProbeAnalysisConfig(seed=11, permutations=24, backend="sklearn", workers=1),
    )
    assert not any(cell["significance"] for cell in result["cells"])
