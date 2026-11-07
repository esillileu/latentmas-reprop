"""Statistical testing utilities for condition outcome comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import binomtest


@dataclass(frozen=True)
class McNemarResult:
    """Results from McNemar statistical comparison between two conditions."""

    n11: int
    n10: int
    n01: int
    n00: int
    accuracy_diff: float
    discordant_count: int
    p_value: float
    statistic: float | None = None

    @property
    def contingency_table(self) -> dict[str, int]:
        """Return 2x2 contingency table representation."""
        return {
            "n11": self.n11,
            "n10": self.n10,
            "n01": self.n01,
            "n00": self.n00,
        }


def mcnemar(a: Any, b: Any) -> McNemarResult:
    """Perform exact McNemar test comparing outcomes of condition a and condition b.

    Parameters:
        a: ConditionOutcome or compatible object with sample_ids and correct attributes.
        b: ConditionOutcome or compatible object with sample_ids and correct attributes.

    Returns:
        McNemarResult containing contingency table, accuracy difference, discordant count, and p-value.

    Raises:
        ValueError: If datasets, splits, or golds do not match, or sample IDs differ.
    """
    # Validate datasets / tasks
    dataset_a = getattr(a, "dataset", None)
    dataset_b = getattr(b, "dataset", None)
    if dataset_a is not None and dataset_b is not None and dataset_a != dataset_b:
        raise ValueError(f"Dataset mismatch: '{dataset_a}' vs '{dataset_b}'")

    # Validate split
    split_a = getattr(a, "split", None)
    split_b = getattr(b, "split", None)
    if split_a is not None and split_b is not None and split_a != split_b:
        raise ValueError(f"Split mismatch: '{split_a}' vs '{split_b}'")

    # Build mapping for samples
    ids_a = list(a.sample_ids)
    corr_a = list(a.correct)
    ids_b = list(b.sample_ids)
    corr_b = list(b.correct)

    if len(ids_a) != len(set(ids_a)):
        dups_a = {x for x in ids_a if ids_a.count(x) > 1}
        raise ValueError(
            f"Duplicate sample IDs in condition a: {sorted(dups_a, key=str)}"
        )
    if len(ids_b) != len(set(ids_b)):
        dups_b = {x for x in ids_b if ids_b.count(x) > 1}
        raise ValueError(
            f"Duplicate sample IDs in condition b: {sorted(dups_b, key=str)}"
        )

    if len(ids_a) != len(corr_a):
        raise ValueError(
            "Length mismatch between sample_ids and correct vector in condition a"
        )
    if len(ids_b) != len(corr_b):
        raise ValueError(
            "Length mismatch between sample_ids and correct vector in condition b"
        )

    map_a = dict(zip(ids_a, corr_a, strict=True))
    map_b = dict(zip(ids_b, corr_b, strict=True))

    set_a = set(ids_a)
    set_b = set(ids_b)

    if set_a != set_b:
        diff_a = set_a - set_b
        diff_b = set_b - set_a
        raise ValueError(
            f"Sample ID sets do not match between conditions. Extra in A: {sorted(diff_a, key=str)[:5]}, Extra in B: {sorted(diff_b, key=str)[:5]}"
        )

    # Validate golds
    golds_a = getattr(a, "golds", None)
    golds_b = getattr(b, "golds", None)
    if golds_a and golds_b:
        if len(golds_a) != len(ids_a):
            raise ValueError(
                "Length mismatch between sample_ids and golds in condition a"
            )
        if len(golds_b) != len(ids_b):
            raise ValueError(
                "Length mismatch between sample_ids and golds in condition b"
            )
        gold_map_a = dict(zip(ids_a, golds_a, strict=True))
        gold_map_b = dict(zip(ids_b, golds_b, strict=True))
        for sid in set_a:
            ga = gold_map_a[sid]
            gb = gold_map_b[sid]
            if ga is not None and gb is not None and ga != gb:
                raise ValueError(
                    f"Gold answer mismatch for sample ID {sid}: '{ga}' vs '{gb}'"
                )

    # Align samples by sorted sample ID
    def _sort_key(sid: int | str) -> tuple[int, int | str]:
        if isinstance(sid, int):
            return (0, sid)
        if isinstance(sid, str) and sid.isdigit():
            return (0, int(sid))
        return (1, str(sid))

    aligned_ids = sorted(set_a, key=_sort_key)

    n11 = 0
    n10 = 0
    n01 = 0
    n00 = 0

    for sid in aligned_ids:
        ca = bool(map_a[sid])
        cb = bool(map_b[sid])
        if ca and cb:
            n11 += 1
        elif ca and not cb:
            n10 += 1
        elif not ca and cb:
            n01 += 1
        else:
            n00 += 1

    discordant = n10 + n01
    total_samples = len(aligned_ids)
    acc_a = (
        sum(1 for sid in aligned_ids if map_a[sid]) / total_samples
        if total_samples > 0
        else 0.0
    )
    acc_b = (
        sum(1 for sid in aligned_ids if map_b[sid]) / total_samples
        if total_samples > 0
        else 0.0
    )
    accuracy_diff = acc_a - acc_b

    # Exact two-sided binomial test
    if discordant == 0:
        p_val = 1.0
        stat = 0.0
    else:
        test_res = binomtest(k=n10, n=discordant, p=0.5, alternative="two-sided")
        p_val = float(test_res.pvalue)
        # Asymptotic continuity-corrected chi-squared statistic for reference
        stat = (
            ((abs(n10 - n01) - 1.0) ** 2) / discordant if abs(n10 - n01) >= 1.0 else 0.0
        )

    return McNemarResult(
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        accuracy_diff=accuracy_diff,
        discordant_count=discordant,
        p_value=p_val,
        statistic=stat,
    )


def paired_uncertainty(result: McNemarResult, confidence: float = 0.95) -> dict:
    """Conservative exact CI for P(A correct)-P(B correct).

    Bound both discordant cell probabilities with Clopper-Pearson intervals
    at Bonferroni-adjusted coverage. Their difference covers the paired effect
    with at least the requested pointwise confidence, including zero discordance.
    This uses items, never repeated steps, as the sampling unit.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one")
    n = result.n11 + result.n10 + result.n01 + result.n00
    fields = {
        "mcnemar_exact_p": None,
        "accuracy_diff_ci_low": None,
        "accuracy_diff_ci_high": None,
        "confidence": confidence,
        "ci_method": "bonferroni_clopper_pearson_discordant_cells",
        "inference_scope": "exploratory_pointwise_selected_items",
    }
    if n:
        positive, negative = [
            binomtest(k, n).proportion_ci(
                confidence_level=1 - (1 - confidence) / 2, method="exact"
            )
            for k in (result.n10, result.n01)
        ]
        fields.update(
            mcnemar_exact_p=result.p_value,
            accuracy_diff_ci_low=max(-1.0, positive.low - negative.high),
            accuracy_diff_ci_high=min(1.0, positive.high - negative.low),
        )
    return fields
