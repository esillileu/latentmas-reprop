"""Statistical computation and metrics aggregation for intervention experiments."""

from typing import Any

from ...domain.models import InterventionMetrics, SampleInterventionRecord


def generate_cross_indices(
    n: int,
    policy: str = "shift_1",
    seed: int = 42,
    cache_seq_lens: list[int] | None = None,
) -> list[int]:
    """Re-export generate_cross_indices from algo module for compatibility."""
    from .algo import generate_cross_indices as _impl

    return _impl(n=n, policy=policy, seed=seed, cache_seq_lens=cache_seq_lens)


def compute_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Compute McNemar test statistic with continuity correction and exact binomial p-value."""
    n = b + c
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "b_own_correct_other_wrong": b,
            "c_own_wrong_other_correct": c,
            "discordant_pairs": 0,
        }
    diff = abs(b - c)
    stat = ((diff - 1) ** 2) / n if diff >= 1 else 0.0
    try:
        from scipy.stats import binomtest

        p_val = binomtest(k=b, n=n, p=0.5).pvalue
    except Exception:
        p_val = 1.0
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_val), 4),
        "b_own_correct_other_wrong": b,
        "c_own_wrong_other_correct": c,
        "discordant_pairs": n,
    }


def compute_cache_and_ram_stats(
    cache_bytes_list: list[int],
    context_seq_lens: list[int],
    cross_indices: list[int],
    conditions: list[str],
    n_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute RAM usage and sequence length statistics for intervention caches."""
    ram_stats: dict[str, Any] = {}
    if cache_bytes_list:
        avg_bytes = sum(cache_bytes_list) / len(cache_bytes_list)
        ram_stats = {
            "avg_cache_bytes_per_sample": int(avg_bytes),
            "max_cache_bytes_per_sample": max(cache_bytes_list),
            "estimated_ram_100_samples_mb": round(avg_bytes * 100 / (1024**2), 2),
            "estimated_ram_300_samples_mb": round(avg_bytes * 300 / (1024**2), 2),
            "n_samples_measured": n_samples,
        }

    cache_seq_stats: dict[str, Any] = {}
    if context_seq_lens:
        csl = context_seq_lens
        cache_seq_stats = {
            "own_seq_len_mean": round(sum(csl) / len(csl), 2),
            "own_seq_len_min": min(csl),
            "own_seq_len_max": max(csl),
        }
        if "cross" in conditions and cross_indices:
            cross_src_lens = [
                context_seq_lens[cross_indices[ii]] for ii in range(n_samples)
            ]
            deltas = [abs(csl[ii] - cross_src_lens[ii]) for ii in range(n_samples)]
            cache_seq_stats.update(
                {
                    "cross_src_seq_len_mean": round(
                        sum(cross_src_lens) / len(cross_src_lens), 2
                    ),
                    "cross_seq_len_delta_mean": round(sum(deltas) / len(deltas), 2),
                    "cross_seq_len_delta_max": max(deltas),
                }
            )
    return ram_stats, cache_seq_stats


def calculate_intervention_metrics(
    sample_records: list[SampleInterventionRecord],
    conditions: list[str],
    n_samples: int,
    total_runtime: float,
    condition_latencies: dict[str, list[float]],
    condition_tokens: dict[str, list[int]],
    cache_seq_stats: dict[str, Any],
    ram_stats: dict[str, Any],
    max_tok: int,
) -> tuple[
    InterventionMetrics,
    dict[str, Any],
    dict[str, dict[str, int]],
    dict[str, Any],
    dict[str, int],
]:
    """Aggregate per-condition and paired transition metrics across intervention records."""
    results_by_sample_and_cond: list[dict[str, SampleInterventionRecord]] = [
        {} for _ in range(n_samples)
    ]
    for r in sample_records:
        idx = (
            r.sample_id
            if isinstance(r.sample_id, int)
            else int(r.sample_id)
            if isinstance(r.sample_id, str) and r.sample_id.isdigit()
            else r.sample_index
        )
        if 0 <= idx < n_samples:
            results_by_sample_and_cond[idx][r.condition] = r

    n_correct_by_cond = {
        cond: sum(
            1
            for ii in range(n_samples)
            if results_by_sample_and_cond[ii].get(cond)
            and results_by_sample_and_cond[ii][cond].correct
        )
        for cond in conditions
    }
    acc_by_cond = {
        cond: n_correct_by_cond[cond] / n_samples if n_samples > 0 else 0.0
        for cond in conditions
    }
    acc_own = acc_by_cond.get("own", 0.0)

    change_rate_by_cond: dict[str, float] = {}
    for cond in conditions:
        if cond == "own":
            continue
        changed = sum(
            1
            for ii in range(n_samples)
            if results_by_sample_and_cond[ii].get(cond)
            and results_by_sample_and_cond[ii].get("own")
            and results_by_sample_and_cond[ii][cond].prediction
            != results_by_sample_and_cond[ii]["own"].prediction
        )
        change_rate_by_cond[cond] = changed / n_samples if n_samples > 0 else 0.0

    paired_transitions: dict[str, Any] = {}
    harm_rescue: dict[str, dict[str, int]] = {}
    mcnemar_dict: dict[str, Any] = {}

    for cond in conditions:
        if cond == "own":
            continue
        c_c = c_w = w_c = w_w = 0
        for ii in range(n_samples):
            own_r = results_by_sample_and_cond[ii].get("own")
            cond_r = results_by_sample_and_cond[ii].get(cond)
            if own_r and cond_r:
                if own_r.correct and cond_r.correct:
                    c_c += 1
                elif own_r.correct and not cond_r.correct:
                    c_w += 1
                elif not own_r.correct and cond_r.correct:
                    w_c += 1
                else:
                    w_w += 1
        paired_transitions[f"own_vs_{cond}"] = {
            "own_correct_cond_correct": c_c,
            "own_correct_cond_wrong": c_w,
            "own_wrong_cond_correct": w_c,
            "own_wrong_cond_wrong": w_w,
        }
        harm_rescue[cond] = {"harm": c_w, "rescue": w_c}
        mcnemar_dict[f"own_vs_{cond}"] = compute_mcnemar_test(b=c_w, c=w_c)

    correctness_patterns: dict[str, int] = {}
    for ii in range(n_samples):
        pattern_parts = [
            f"{cond}={1 if (results_by_sample_and_cond[ii].get(cond) and results_by_sample_and_cond[ii][cond].correct) else 0}"
            for cond in conditions
        ]
        pattern = " ".join(pattern_parts)
        correctness_patterns[pattern] = correctness_patterns.get(pattern, 0) + 1

    runtime_by_cond = {
        cond: sum(condition_latencies.get(cond, [])) for cond in conditions
    }
    tokens_mean: dict[str, float] = {}
    tokens_max: dict[str, int] = {}
    truncation_rates: dict[str, float] = {}
    for cond in conditions:
        toks = condition_tokens.get(cond, [])
        if toks:
            tokens_mean[cond] = round(sum(toks) / len(toks), 2)
            tokens_max[cond] = max(toks)
            truncation_rates[cond] = (
                round(sum(1 for t in toks if t >= max_tok) / len(toks), 4)
                if max_tok > 0
                else 0.0
            )

    runtime_per_sample = total_runtime / n_samples if n_samples > 0 else 0.0
    metrics = InterventionMetrics(
        accuracy_own=round(acc_by_cond.get("own", 0.0), 4),
        accuracy_cross=round(acc_by_cond.get("cross", 0.0), 4),
        accuracy_drop=round(acc_by_cond.get("drop", 0.0), 4),
        accuracy_zero=round(acc_by_cond.get("zero", 0.0), 4),
        accuracy_by_condition={c: round(v, 4) for c, v in acc_by_cond.items()},
        n_correct_by_condition=n_correct_by_cond,
        answer_change_rate_cross=round(change_rate_by_cond.get("cross", 0.0), 4),
        answer_change_rate_drop=round(change_rate_by_cond.get("drop", 0.0), 4),
        answer_change_rate_zero=round(change_rate_by_cond.get("zero", 0.0), 4),
        answer_change_rate_by_condition={
            c: round(v, 4) for c, v in change_rate_by_cond.items()
        },
        accuracy_delta_own_cross=round(acc_own - acc_by_cond.get("cross", 0.0), 4),
        accuracy_delta_own_drop=round(acc_own - acc_by_cond.get("drop", 0.0), 4),
        accuracy_delta_own_zero=round(acc_own - acc_by_cond.get("zero", 0.0), 4),
        n_samples=n_samples,
        n_own_correct=n_correct_by_cond.get("own", 0),
        n_cross_correct=n_correct_by_cond.get("cross", 0),
        n_drop_correct=n_correct_by_cond.get("drop", 0),
        n_zero_correct=n_correct_by_cond.get("zero", 0),
        runtime_total=round(total_runtime, 4),
        runtime_per_sample=round(runtime_per_sample, 4),
        runtime_own=round(runtime_by_cond.get("own", 0.0), 4),
        runtime_cross=round(runtime_by_cond.get("cross", 0.0), 4),
        runtime_drop=round(runtime_by_cond.get("drop", 0.0), 4),
        runtime_zero=round(runtime_by_cond.get("zero", 0.0), 4),
        runtime_by_condition={c: round(v, 4) for c, v in runtime_by_cond.items()},
        paired_transitions=paired_transitions,
        harm_rescue=harm_rescue,
        correctness_patterns=correctness_patterns,
        mcnemar_tests=mcnemar_dict,
        cache_seq_len_stats=cache_seq_stats,
        ram_usage_stats=ram_stats,
        max_new_tokens=max_tok,
        tokens_generated_mean=tokens_mean,
        tokens_generated_max=tokens_max,
        truncation_rate=truncation_rates,
    )
    return metrics, paired_transitions, harm_rescue, mcnemar_dict, correctness_patterns
