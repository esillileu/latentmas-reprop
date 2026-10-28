"""Digit distribution statistics from saved observations."""

import numpy as np

DIGITS = range(10)


def entropy(probabilities):
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def mutual_information(labels, predictions):
    matrix = np.zeros((10, 10))
    np.add.at(matrix, (labels, predictions), 1)
    joint = matrix / len(labels)
    return (
        entropy(joint.sum(axis=0)) + entropy(joint.sum(axis=1)) - entropy(joint.ravel())
    )


def average(values):
    return np.mean(values, axis=0).tolist() if values else None


def probability_metrics(record, label):
    saved = record.get("candidate_probabilities", {})
    if not all(str(d) in saved for d in DIGITS):
        return None
    p = np.array([saved[str(d)] for d in DIGITS], dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        return None
    q = p / p.sum()
    target = np.eye(10)[label]
    saved_log = record.get("candidate_log_probabilities", {}).get(str(label))
    nll = (
        float(-saved_log + np.log(p.sum()))
        if saved_log is not None and np.isfinite(saved_log)
        else (float(-np.log(q[label])) if q[label] > 0 else None)
    )
    return {
        "digit_probability_mass": float(p.sum()),
        "normalized_digit_probabilities": q.tolist(),
        "source_label_nll": nll,
        "brier_score": float(np.sum((q - target) ** 2)),
    }
