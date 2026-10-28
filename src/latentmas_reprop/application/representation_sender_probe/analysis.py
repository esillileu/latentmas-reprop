"""Grouped out-of-fold probe analysis with max-statistic correction."""

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .estimator import EstimatorConfig, fit_predict, resolve_backend
from .folds import grouped_folds, permute_within_templates
from .torch_solver import TorchSolverConfig, analyze_cells_batched

_THREAD_LIMIT = None
_WORKER_CONTEXT = None


@dataclass(frozen=True)
class ProbeAnalysisConfig:
    seed: int = 42
    folds: int = 5
    permutations: int = 5000
    backend: str = "auto"
    workers: int = 0
    c: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4
    batch_size: int = 256


def _oof(
    features: np.ndarray,
    labels: np.ndarray,
    folds: list[dict[str, list[int]]],
    estimator: EstimatorConfig,
    backend: str,
) -> tuple[float, list[float], list[list[int]], list[int]]:
    predictions = np.empty_like(labels)
    fold_accuracies = []
    for fold in folds:
        train = np.asarray(fold["train_indices"])
        test = np.asarray(fold["test_indices"])
        predictions[test] = fit_predict(
            features[train], labels[train], features[test], estimator, backend
        )
        fold_accuracies.append(float(np.mean(predictions[test] == labels[test])))
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return (
        float(np.mean(predictions == labels)),
        fold_accuracies,
        confusion.tolist(),
        predictions.tolist(),
    )


def _permutation_job(index: int) -> tuple[int, list[float]]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("permutation worker was not initialized")
    seed, labels, templates, cells, folds, estimator, backend = _WORKER_CONTEXT
    derived_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
    permuted = permute_within_templates(labels, templates, seed=derived_seed)
    return index, [_oof(cell, permuted, folds, estimator, backend)[0] for cell in cells]


def _initialize_worker(
    backend: str,
    context: tuple,
) -> None:
    """Keep CPU workers single-threaded and bind each GPU worker to one device."""
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    global _THREAD_LIMIT, _WORKER_CONTEXT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _WORKER_CONTEXT = context


def analyze_sender_states(
    payload: dict[str, Any], config: ProbeAnalysisConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = (
        torch.as_tensor(
            payload.get("labels", [item["digit"] for item in payload["metadata"]])
        )
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    templates = (
        torch.as_tensor(
            payload.get(
                "template_indices",
                [item["template_index"] for item in payload["metadata"]],
            )
        )
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    folds = grouped_folds(templates, seed=config.seed, fold_count=config.folds)
    backend = resolve_backend(config.backend)
    if backend == "sklearn":
        _initialize_worker(backend, ())
    estimator = EstimatorConfig(backend, config.c, config.max_iter, config.tol)
    cell_specs: list[tuple[str, int]] = []
    cell_features: list[np.ndarray] = []
    for representation in ("hidden_pre_realign", "latent_post_realign"):
        tensor = torch.as_tensor(payload[representation]).float().cpu().numpy()
        for step in range(tensor.shape[1]):
            cell_specs.append((representation, step + 1))
            cell_features.append(tensor[:, step, :])
    permuted_labels = (
        np.stack(
            [
                permute_within_templates(
                    labels,
                    templates,
                    seed=int(
                        np.random.SeedSequence([config.seed, index]).generate_state(1)[
                            0
                        ]
                    ),
                )
                for index in range(config.permutations)
            ],
            axis=0,
        )
        if config.permutations
        else np.empty((0, len(labels)), dtype=np.int64)
    )
    if backend == "torch":
        workers = 1
        observed, null, resolved_batch_size = analyze_cells_batched(
            cell_features,
            labels,
            permuted_labels,
            folds,
            TorchSolverConfig(
                c=config.c,
                max_iter=config.max_iter,
                tol=config.tol,
                batch_size=config.batch_size,
            ),
        )
    else:
        resolved_batch_size = None
        observed = [
            _oof(features, labels, folds, estimator, backend)
            for features in cell_features
        ]
        workers = max(1, config.workers or min(8, os.cpu_count() or 1))
        context = (
            config.seed,
            labels,
            templates,
            cell_features,
            folds,
            estimator,
            backend,
        )
        if workers == 1 or config.permutations == 0:
            global _WORKER_CONTEXT
            _WORKER_CONTEXT = context
            permuted = [_permutation_job(index) for index in range(config.permutations)]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_initialize_worker,
                initargs=(backend, context),
            ) as executor:
                chunksize = max(1, config.permutations // (workers * 8))
                permuted = list(
                    executor.map(
                        _permutation_job,
                        range(config.permutations),
                        chunksize=chunksize,
                    )
                )
        permuted.sort(key=lambda item: item[0])
        null = np.asarray([values for _, values in permuted], dtype=float).reshape(
            config.permutations, len(cell_specs)
        )
    null_max = null.max(axis=1) if config.permutations else np.empty(0)
    cells = []
    for cell_index, (
        (representation, step),
        (accuracy, fold_accuracies, confusion, _predictions),
    ) in enumerate(zip(cell_specs, observed, strict=True)):
        raw_extreme = int(np.count_nonzero(null[:, cell_index] >= accuracy))
        corrected_extreme = int(np.count_nonzero(null_max >= accuracy))
        denominator = config.permutations + 1
        raw_p = (raw_extreme + 1) / denominator
        corrected_p = (corrected_extreme + 1) / denominator
        cells.append(
            {
                "representation": representation,
                "step": step,
                "observed_oof_accuracy": accuracy,
                "chance": 0.1,
                "delta": accuracy - 0.1,
                "raw_p_value": raw_p,
                "fwer_p_value": corrected_p,
                "significance": corrected_p < 0.05,
                "fold_accuracies": fold_accuracies,
                "confusion_matrix": confusion,
            }
        )
    solver = (
        "torch_batched_LBFGS"
        if backend == "torch"
        else "LogisticRegression(solver=lbfgs)"
    )
    results = {
        "schema_version": 1,
        "cells": cells,
        "analysis": {
            **asdict(config),
            "resolved_backend": backend,
            "resolved_workers": workers,
            "solver": solver,
            "resolved_batch_size": resolved_batch_size,
        },
    }
    statistics = {
        "folds": folds,
        "cell_order": [
            {"representation": name, "step": step} for name, step in cell_specs
        ],
        "observed_oof_predictions": [item[3] for item in observed],
        "permutation_accuracies": null.tolist(),
        "max_accuracies": null_max.tolist(),
    }
    return results, statistics
