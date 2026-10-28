"""Multinomial L2 logistic-regression estimator backends."""

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class EstimatorConfig:
    backend: str = "auto"
    c: float = 1.0
    max_iter: int = 1000
    tol: float = 1e-4


def resolve_backend(requested: str) -> str:
    if requested not in {"auto", "torch", "sklearn"}:
        raise ValueError("backend must be auto, torch, or sklearn")
    if requested == "torch" or (requested == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch GPU backend requested but CUDA is unavailable")
        return "torch"
    return "sklearn"


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: EstimatorConfig,
    backend: str,
) -> np.ndarray:
    center = train_x.mean(axis=0, keepdims=True)
    scale = train_x.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    train_x = (train_x - center) / scale
    test_x = (test_x - center) / scale
    model = LogisticRegression(
        C=config.c,
        solver="lbfgs",
        max_iter=config.max_iter,
        tol=config.tol,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(train_x, train_y)
        except ConvergenceWarning as error:
            raise RuntimeError(
                "sklearn logistic regression failed to converge"
            ) from error
    return model.predict(test_x).astype(np.int64)
