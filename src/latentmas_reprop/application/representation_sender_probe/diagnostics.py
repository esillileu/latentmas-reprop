"""Grouped probe robustness and geometry from saved Sender features."""

import warnings

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def feature_geometry(features, labels):
    x = np.asarray(features, dtype=np.float64)
    centered = x - x.mean(axis=0)
    digits = np.unique(labels)
    centroids = np.array([x[labels == d].mean(axis=0) for d in digits])
    counts = np.array([np.sum(labels == d) for d in digits])
    within = float(
        sum(
            np.sum((x[labels == d] - c) ** 2)
            for d, c in zip(digits, centroids, strict=True)
        )
    )
    between = float(np.sum(counts[:, None] * (centroids - x.mean(axis=0)) ** 2))
    distances = np.linalg.norm(centroids[:, None] - centroids[None, :], axis=-1)
    separation = distances[np.triu_indices(len(digits), k=1)]
    eigenvalues = (
        np.linalg.eigvalsh(
            (centered.T @ centered if x.shape[1] < len(x) else centered @ centered.T)
            / (len(x) - 1)
        )
        if len(x) > 1
        else np.array([])
    )
    eigenvalues = np.maximum(eigenvalues, 0)[::-1]
    positive = eigenvalues[
        eigenvalues > (eigenvalues.max() * 1e-12 if eigenvalues.size else 0)
    ]
    weights = positive / positive.sum() if positive.size else np.array([])
    return {
        "sample_count": len(x),
        "feature_dimensions": x.shape[1],
        "source_digit_counts": counts.tolist(),
        "source_digits": digits.tolist(),
        "class_centroid_distances": distances.tolist(),
        "centroid_separation_mean": float(separation.mean())
        if separation.size
        else None,
        "centroid_separation_min": float(separation.min()) if separation.size else None,
        "within_class_scatter": within,
        "between_class_scatter": between,
        "between_within_ratio": between / within if within else None,
        "covariance_eigenvalues": eigenvalues.tolist(),
        "covariance_trace": float(eigenvalues.sum()),
        "covariance_rank": len(positive),
        "effective_rank": float(np.exp(-np.sum(weights * np.log(weights))))
        if weights.size
        else 0.0,
        "participation_ratio": float(positive.sum() ** 2 / np.sum(positive**2))
        if positive.size
        else 0.0,
    }


def effect_intervals(payload, statistics, cells, *, seed, bootstrap_count):
    labels = torch.as_tensor(payload["labels"]).cpu().numpy()
    templates = torch.as_tensor(payload["template_indices"]).cpu().numpy()
    predictions = statistics.get("observed_oof_predictions")
    for cell in cells:
        cell.update(effect_ci_low=None, effect_ci_high=None)
    if predictions is None:
        return
    correct = np.asarray(predictions) == labels[None, :]
    groups = np.unique(templates)
    successes = np.array([correct[:, templates == t].sum(axis=1) for t in groups])
    sizes = np.array([np.sum(templates == t) for t in groups])
    weights = np.random.default_rng(seed).multinomial(
        len(groups), np.full(len(groups), 1 / len(groups)), size=bootstrap_count
    )
    accuracies = (weights @ successes) / (weights @ sizes)[:, None]
    for index, cell in enumerate(cells):
        if cell["null_mean_accuracy"] is None:
            continue
        effects = accuracies[:, index] - cell["null_mean_accuracy"]
        low, high = np.quantile(effects, [0.025, 0.975])
        cell.update(
            effect_ci_low=float(low),
            effect_ci_high=float(high),
            effect_ci_method="95% template-cluster percentile bootstrap; fixed OOF predictions and fixed null mean; no refitting",
            bootstrap_count=bootstrap_count,
            bootstrap_seed=seed,
            bootstrap_unit="prompt template",
        )


def robustness_probes(features, labels, folds, *, seed):
    configurations = [("ridge", True, alpha) for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)]
    configurations += [
        ("logistic", False, 1.0),
        ("logistic", True, 1.0),
        ("lda", True, None),
        ("knn", True, 5),
    ]
    results = []
    for name, standardize, parameter in configurations:
        predictions = np.empty_like(labels)
        fold_accuracies, convergence = [], []
        for fold in folds:
            train, test = (
                np.array(fold["train_indices"]),
                np.array(fold["test_indices"]),
            )
            x_train, x_test = features[train], features[test]
            if standardize:
                scaler = StandardScaler().fit(x_train)
                x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
            if name == "ridge":
                estimator = RidgeClassifier(alpha=parameter, solver="svd")
            elif name == "logistic":
                estimator = LogisticRegression(
                    C=parameter, max_iter=1000, tol=1e-4, random_state=seed
                )
            elif name == "lda":
                estimator = LinearDiscriminantAnalysis(solver="svd")
            else:
                estimator = KNeighborsClassifier(n_neighbors=parameter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(x_train, labels[train])
            convergence.append(
                not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            )
            predictions[test] = estimator.predict(x_test)
            fold_accuracies.append(float(np.mean(predictions[test] == labels[test])))
        results.append(
            {
                "probe": name,
                "standardization": standardize,
                "parameter": parameter,
                "accuracy": float(np.mean(predictions == labels)),
                "fold_accuracies": fold_accuracies,
                "fold_converged": convergence,
                "oof_predictions": predictions.tolist(),
                "seed": seed,
                "folds": folds,
            }
        )
    return results


def analyze_saved_features(payload, statistics, cells, *, seed=0, bootstrap_count=5000):
    labels = torch.as_tensor(payload["labels"]).cpu().numpy()
    effect_intervals(
        payload, statistics, cells, seed=seed, bootstrap_count=bootstrap_count
    )
    robustness, geometry = [], []
    with threadpool_limits(limits=1):
        for cell in cells:
            identity = {
                k: cell[k]
                for k in ("model", "latent_steps", "run_id", "representation", "step")
            }
            features = (
                torch.as_tensor(payload[cell["representation"]])[:, cell["step"] - 1]
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            geometry.append({**identity, **feature_geometry(features, labels)})
            robustness.extend(
                {**identity, **row}
                for row in robustness_probes(
                    features, labels, statistics["folds"], seed=seed
                )
            )
    return robustness, geometry
