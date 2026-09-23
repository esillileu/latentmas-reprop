"""Tests for grouped sender-state probe analysis."""

import numpy as np
import torch

from latentmas_reprop.application.sender_probe import (
    ProbeAnalysisConfig,
    analyze_sender_states,
)
from latentmas_reprop.application.sender_probe.folds import grouped_folds
from latentmas_reprop.application.sender_probe.replacement import (
    LEGACY_PROBE_PARAMS,
    _analysis_params,
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
