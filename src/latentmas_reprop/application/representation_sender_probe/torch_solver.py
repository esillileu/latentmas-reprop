"""Batched GPU multinomial logistic regression for permutation probes."""

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class TorchSolverConfig:
    c: float
    max_iter: int
    tol: float
    batch_size: int
    device: str = "cuda"
    history_size: int = 10


def _fit_predict(
    train_x: torch.Tensor,
    train_labels: torch.Tensor,
    test_x: torch.Tensor,
    config: TorchSolverConfig,
) -> torch.Tensor:
    """Fit independent classifiers for a batch of label vectors."""
    batch_size, sample_count = train_labels.shape
    feature_count = train_x.shape[1]
    weights = torch.zeros(
        batch_size,
        feature_count,
        10,
        device=train_x.device,
        requires_grad=True,
    )
    intercept = torch.zeros(
        batch_size, 1, 10, device=train_x.device, requires_grad=True
    )
    penalty_scale = 1.0 / (2.0 * config.c * sample_count)
    optimizer = torch.optim.LBFGS(
        [weights, intercept],
        max_iter=config.max_iter,
        tolerance_grad=config.tol,
        tolerance_change=config.tol,
        history_size=config.history_size,
        line_search_fn="strong_wolfe",
    )

    def objective(*, backward: bool) -> torch.Tensor:
        logits = torch.einsum("nd,bdk->bnk", train_x, weights) + intercept
        losses = functional.cross_entropy(
            logits.transpose(1, 2), train_labels, reduction="none"
        ).mean(dim=1)
        losses = losses + penalty_scale * weights.square().sum(dim=(1, 2))
        total = losses.sum()
        if backward:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
        return total

    def closure() -> torch.Tensor:
        return objective(backward=True)

    optimizer.step(closure)
    objective(backward=True)
    maximum_gradient = max(
        float(weights.grad.abs().amax().item()),
        float(intercept.grad.abs().amax().item()),
    )
    iterations = int(optimizer.state[weights].get("n_iter", config.max_iter))
    if iterations >= config.max_iter and maximum_gradient > config.tol:
        raise RuntimeError(
            "batched logistic regression failed to converge: "
            f"maximum gradient {maximum_gradient:.6g} exceeds {config.tol:.6g} "
            f"after {iterations} iterations"
        )
    with torch.no_grad():
        logits = torch.einsum("md,bdk->bmk", test_x, weights) + intercept
        return logits.argmax(dim=-1)


def analyze_cells_batched(
    cell_features: list[np.ndarray],
    labels: np.ndarray,
    permuted_labels: np.ndarray,
    folds: list[dict[str, list[int]]],
    config: TorchSolverConfig,
) -> tuple[
    list[tuple[float, list[float], list[list[int]], list[int]]], np.ndarray, int
]:
    """Fit observed and batched permutation probes for every cell."""
    device = torch.device(config.device)
    labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
    null = np.empty((len(permuted_labels), len(cell_features)), dtype=np.float64)
    observed = []
    effective_batch_size = min(config.batch_size, max(1, len(permuted_labels)))
    for cell_index, features in enumerate(cell_features):
        cell_started = time.perf_counter()
        print(
            f"Probe cell {cell_index + 1}/{len(cell_features)}: observed fit",
            flush=True,
        )
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        prepared_folds = []
        for fold in folds:
            train = torch.as_tensor(fold["train_indices"], device=device)
            test = torch.as_tensor(fold["test_indices"], device=device)
            train_x = feature_tensor[train]
            center = train_x.mean(dim=0, keepdim=True)
            scale = train_x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            prepared_folds.append(
                (
                    (train_x - center) / scale,
                    (feature_tensor[test] - center) / scale,
                    train,
                    test,
                )
            )

        predictions = torch.empty_like(labels_tensor)
        fold_accuracies = []
        for train_x, test_x, train, test in prepared_folds:
            fold_predictions = _fit_predict(
                train_x,
                labels_tensor[train].unsqueeze(0),
                test_x,
                config,
            )[0]
            predictions[test] = fold_predictions
            fold_accuracies.append(
                float((fold_predictions == labels_tensor[test]).float().mean().item())
            )
        expected = labels_tensor.cpu().numpy()
        predicted = predictions.cpu().numpy()
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (expected, predicted), 1)
        observed.append(
            (
                float(np.mean(predicted == expected)),
                fold_accuracies,
                confusion.tolist(),
                predicted.tolist(),
            )
        )

        start = 0
        while start < len(permuted_labels):
            stop = min(start + effective_batch_size, len(permuted_labels))
            batch_labels = torch.as_tensor(
                permuted_labels[start:stop], dtype=torch.long, device=device
            )
            try:
                correct = torch.zeros(stop - start, dtype=torch.long, device=device)
                for train_x, test_x, train, test in prepared_folds:
                    fold_predictions = _fit_predict(
                        train_x,
                        batch_labels[:, train],
                        test_x,
                        config,
                    )
                    correct += (fold_predictions == batch_labels[:, test]).sum(dim=1)
            except torch.cuda.OutOfMemoryError:
                del batch_labels
                torch.cuda.empty_cache()
                if effective_batch_size == 1:
                    raise
                effective_batch_size = max(1, effective_batch_size // 2)
                print(
                    f"CUDA OOM; retrying with batch size {effective_batch_size}",
                    flush=True,
                )
                continue
            null[start:stop, cell_index] = (
                correct.float().div(len(labels)).cpu().numpy()
            )
            start = stop
            elapsed = time.perf_counter() - cell_started
            rate = start / elapsed
            eta = (len(permuted_labels) - start) / rate if rate else 0.0
            print(
                f"Probe cell {cell_index + 1}/{len(cell_features)}: "
                f"permutations {start}/{len(permuted_labels)}, "
                f"elapsed {elapsed:.1f}s, ETA {eta:.1f}s",
                flush=True,
            )
    return observed, null, effective_batch_size
