"""Deterministic template-grouped folds and label permutations."""

import numpy as np


def grouped_folds(
    template_indices: np.ndarray, *, seed: int, fold_count: int = 5
) -> list[dict[str, list[int]]]:
    templates = np.unique(template_indices)
    if len(templates) < fold_count:
        raise ValueError(f"at least {fold_count} templates are required")
    shuffled = np.random.default_rng(seed).permutation(templates)
    test_groups = [shuffled[index::fold_count] for index in range(fold_count)]
    folds = []
    all_indices = np.arange(len(template_indices))
    for group in test_groups:
        test = all_indices[np.isin(template_indices, group)]
        train = all_indices[~np.isin(template_indices, group)]
        folds.append(
            {
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "train_template_indices": sorted(set(template_indices[train].tolist())),
                "test_template_indices": sorted(set(template_indices[test].tolist())),
            }
        )
    return folds


def permute_within_templates(
    labels: np.ndarray, template_indices: np.ndarray, *, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    result = labels.copy()
    for template in np.unique(template_indices):
        indices = np.flatnonzero(template_indices == template)
        result[indices] = rng.permutation(labels[indices])
    return result
