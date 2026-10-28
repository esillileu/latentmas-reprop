"""Label-permutation tests and probability-level observations from saved rows."""

import numpy as np

from .statistics import average, probability_metrics


def permutation_result(observed, null, *, lower, seed, count, sample_count):
    values = np.asarray(null)
    tolerance = 100 * np.finfo(float).eps * max(1.0, abs(observed))
    extreme = (
        values <= observed + tolerance if lower else values >= observed - tolerance
    )
    return {
        "observed": observed,
        "null_mean": float(values.mean()) if values.size else None,
        "raw_p_value": (1 + int(extreme.sum())) / (len(values) + 1)
        if values.size
        else None,
        "alternative": "less" if lower else "greater",
        "permutation_statistics": values.tolist(),
        "permutations": count,
        "seed": seed,
        "sample_count": sample_count,
        "shuffle_unit": "individual sample source labels within run and condition",
    }


def balanced_accuracy(labels, predictions):
    return float(
        np.mean(
            [
                np.mean(predictions[labels == digit] == digit)
                for digit in np.unique(labels)
            ]
        )
    )


def analyze_probability_records(group, *, drop, permutations, seed):
    labels = np.array([r["target_digit"] if drop else r["source_digit"] for r in group])
    predictions = np.array([r["predicted_digit"] for r in group])
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    observed = balanced_accuracy(labels, predictions)
    null = [
        balanced_accuracy(rng.permutation(labels), predictions)
        for _ in range(permutations)
    ]
    recovery = permutation_result(
        observed,
        null,
        lower=False,
        seed=seed,
        count=permutations,
        sample_count=len(group),
    )
    recovery["rng_stream"] = 1
    samples, nll_rows, brier_rows = [], [], []
    for r, label in zip(group, labels, strict=True):
        metrics = probability_metrics(r, int(label))
        row = {
            "sample_index": r["sample_index"],
            "source_digit": int(label),
            "correct_probability": r.get("candidate_probabilities", {}).get(str(label)),
            "normalized_correct_probability": None,
            "correct_digit_rank": None,
            "top_2": None,
            "top_3": None,
            "correct_digit_margin": None,
            "source_label_nll": None,
            "brier_score": None,
        }
        if metrics is not None:
            q = np.array(metrics["normalized_digit_probabilities"])
            order = np.argsort(-q, kind="stable")
            rank = int(np.flatnonzero(order == label)[0]) + 1
            row.update(
                normalized_correct_probability=float(q[label]),
                correct_digit_rank=rank,
                top_2=rank <= 2,
                top_3=rank <= 3,
                correct_digit_margin=float(q[label] - np.max(np.delete(q, label))),
                source_label_nll=metrics["source_label_nll"],
                brier_score=metrics["brier_score"],
            )
            nll_rows.append(
                [probability_metrics(r, d)["source_label_nll"] for d in range(10)]
            )
            brier_rows.append(
                [probability_metrics(r, d)["brier_score"] for d in range(10)]
            )
        samples.append(row)
    valid_labels = np.array(
        [
            row["source_digit"]
            for row in samples
            if row["normalized_correct_probability"] is not None
        ]
    )
    tests = {}
    rng = np.random.default_rng(np.random.SeedSequence([seed, 2]))
    shuffles = (
        [rng.permutation(valid_labels) for _ in range(permutations)]
        if len(valid_labels)
        else []
    )
    for name, losses in (("source_label_nll", nll_rows), ("brier_score", brier_rows)):
        # All candidate losses must be available; do not invent a log probability for zeros.
        if not losses or any(value is None for row in losses for value in row):
            tests[name] = None
            continue
        loss = np.array(losses)
        indices = np.arange(len(valid_labels))
        observed_loss = float(loss[indices, valid_labels].mean())
        null_losses = [float(loss[indices, shuffled].mean()) for shuffled in shuffles]
        tests[name] = permutation_result(
            observed_loss,
            null_losses,
            lower=True,
            seed=seed,
            count=permutations,
            sample_count=len(valid_labels),
        )
        tests[name]["rng_stream"] = 2
    return samples, recovery, tests


def enrich_condition(values, group, *, drop, permutations, seed):
    samples, recovery, tests = analyze_probability_records(
        group, drop=drop, permutations=permutations, seed=seed
    )
    values["probability_samples"] = samples
    values["exact_recovery"] = recovery
    values["soft_dependency"] = tests
    metric_names = (
        "correct_probability",
        "normalized_correct_probability",
        "correct_digit_rank",
        "top_2",
        "top_3",
        "correct_digit_margin",
        "source_label_nll",
        "brier_score",
    )
    for key in metric_names:
        values["mean_" + key] = average(
            [row[key] for row in samples if row[key] is not None]
        )
    for digit, stat in enumerate(values["per_digit_accuracy"]):
        rows = [row for row in samples if row["source_digit"] == digit]
        stat["probability_sample_count"] = sum(
            row["normalized_correct_probability"] is not None for row in rows
        )
        for key in metric_names:
            stat["mean_" + key] = average(
                [row[key] for row in rows if row[key] is not None]
            )


def correct_receiver_families(runs):
    """Holm FWER correction separately for MI, NLL, Brier and exact recovery."""
    families = {
        name: []
        for name in ("mi", "source_label_nll", "brier_score", "balanced_accuracy")
    }
    for run in runs:
        for condition, values in run["conditions"].items():
            if condition == "drop":
                continue
            if "dependency" in values:
                families["mi"].append((values["dependency"], "permutation_p_value"))
            for name, test in values.get("soft_dependency", {}).items():
                if test is not None:
                    families[name].append((test, "raw_p_value"))
            if "exact_recovery" in values:
                families["balanced_accuracy"].append(
                    (values["exact_recovery"], "raw_p_value")
                )
    for name, family in families.items():
        ranked = sorted(
            ((test, key) for test, key in family if test[key] is not None),
            key=lambda pair: pair[0][pair[1]],
        )
        adjusted = 0.0
        for index, (test, key) in enumerate(ranked):
            adjusted = max(adjusted, min(1.0, (len(ranked) - index) * test[key]))
            test.update(
                holm_p_value=adjusted,
                correction="Holm",
                family=name,
                family_size=len(ranked),
            )
