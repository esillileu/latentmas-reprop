"""Sender geometry, cluster uncertainty and held-out robustness."""

import numpy as np
import pytest

from latentmas_reprop.application.sender_probe.diagnostics import (
    effect_intervals,
    feature_geometry,
    robustness_probes,
)
from latentmas_reprop.application.sender_probe.folds import grouped_folds


def test_geometry_scatter_and_covariance_spectrum():
    x = np.array([[0, 0], [0, 2], [4, 0], [4, 2]])
    result = feature_geometry(x, np.array([0, 0, 1, 1]))
    assert result["within_class_scatter"] == 4
    assert result["between_class_scatter"] == 16
    assert result["centroid_separation_mean"] == 4
    assert result["covariance_eigenvalues"] == pytest.approx(
        np.linalg.eigvalsh(np.cov(x.T))[::-1]
    )
    assert result["covariance_rank"] == 2
    assert 1 < result["effective_rank"] < 2


def test_effect_ci_resamples_entire_templates():
    payload = {"labels": [0, 1, 0, 1], "template_indices": [0, 0, 1, 1]}
    statistics = {"observed_oof_predictions": [[0, 1, 1, 0]]}
    cells = [{"null_mean_accuracy": 0.1}]
    effect_intervals(payload, statistics, cells, seed=7, bootstrap_count=1000)
    assert cells[0]["effect_ci_low"] == pytest.approx(-0.1)
    assert cells[0]["effect_ci_high"] == pytest.approx(0.9)
    assert cells[0]["bootstrap_unit"] == "prompt template"


def test_all_robustness_presets_use_template_held_out_predictions():
    templates = np.repeat(np.arange(5), 10)
    labels = np.tile(np.arange(10), 5)
    rng = np.random.default_rng(4)
    features = np.eye(10)[labels] * 10 + rng.normal(0, 0.01, (50, 10))
    folds = grouped_folds(templates, seed=2)
    results = robustness_probes(features, labels, folds, seed=2)
    assert len(results) == 9
    assert {row["probe"] for row in results} == {"ridge", "logistic", "lda", "knn"}
    assert {row["parameter"] for row in results if row["probe"] == "ridge"} == {
        0.01,
        0.1,
        1,
        10,
        100,
    }
    for row in results:
        assert len(row["oof_predictions"]) == 50
        assert row["accuracy"] > 0.9
        assert row["folds"] == folds
