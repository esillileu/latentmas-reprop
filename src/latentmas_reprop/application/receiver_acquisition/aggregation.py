"""Result aggregation and metric computation for receiver acquisition experiments."""

from statistics import mean
from typing import Any

from ...domain.models import ReceiverAcquisitionMetrics, ReceiverAcquisitionRecord
from ...domain.services.prompts.acquisition import CANDIDATE_DIGITS


def aggregate_receiver_records(
    records: list[ReceiverAcquisitionRecord],
    n_samples: int,
    expected: int,
    runtime: float,
    context_runtime: float,
) -> ReceiverAcquisitionMetrics:
    """Aggregate per-condition cells, deltas, and stats into ReceiverAcquisitionMetrics."""
    successful = [record for record in records if record.error is None]
    cells: dict[str, dict[str, Any]] = {}
    for cell in sorted(
        {
            f"{r.context_mode}/{r.condition}" if r.condition != "drop" else "drop"
            for r in records
        }
    ):
        group = [
            r
            for r in successful
            if (f"{r.context_mode}/{r.condition}" if r.condition != "drop" else "drop")
            == cell
        ]

        value_lists = {
            attr: [
                getattr(record, attr)
                for record in group
                if getattr(record, attr) is not None
            ]
            for attr in (
                "content_follow_correct",
                "source_probability",
                "source_rank",
                "source_margin",
                "target_probability",
                "target_rank",
                "candidate_mass",
            )
        }

        cells[cell] = {
            "n_successful": len(group),
            "content_follow_accuracy": mean(value_lists["content_follow_correct"])
            if value_lists["content_follow_correct"]
            else None,
            "mean_source_probability": mean(value_lists["source_probability"])
            if value_lists["source_probability"]
            else None,
            "mean_source_rank": mean(value_lists["source_rank"])
            if value_lists["source_rank"]
            else None,
            "mean_source_margin": mean(value_lists["source_margin"])
            if value_lists["source_margin"]
            else None,
            "mean_target_probability": mean(value_lists["target_probability"])
            if value_lists["target_probability"]
            else None,
            "mean_target_rank": mean(value_lists["target_rank"])
            if value_lists["target_rank"]
            else None,
            "mean_candidate_mass": mean(value_lists["candidate_mass"])
            if value_lists["candidate_mass"]
            else None,
            "mean_candidate_probabilities": {
                digit: mean(r.candidate_probabilities[digit] for r in group)
                for digit in CANDIDATE_DIGITS
            }
            if group
            else {},
        }
    drops = {r.sample_index: r for r in successful if r.condition == "drop"}
    deltas: dict[str, float] = {}
    comparable_cells = (
        "full/own",
        "full/cross",
        "prompt_only/own",
        "prompt_only/cross",
        "latent_only/own",
        "latent_only/cross",
    )
    for cell in comparable_cells:
        values = [
            r.source_probability
            - drops[r.sample_index].candidate_probabilities[str(r.source_digit)]
            for r in successful
            if f"{r.context_mode}/{r.condition}" == cell
            and r.source_probability is not None
            and r.sample_index in drops
        ]
        if values:
            deltas[cell] = mean(values)
    cross_follow = {
        mode: cells.get(f"{mode}/cross", {}).get("content_follow_accuracy", 0.0) or 0.0
        for mode in (
            "full",
            "prompt_only",
            "latent_only",
        )
    }
    cross_retention = {}
    for mode in (
        "full",
        "prompt_only",
        "latent_only",
    ):
        vals = [
            r.target_retained
            for r in successful
            if r.context_mode == mode and r.condition == "cross"
        ]
        cross_retention[mode] = mean(vals) if vals else 0.0
    drop_records = list(drops.values())
    carrier_comparison: dict[str, dict[str, float]] = {}
    for carrier in ("full", "prompt_only", "latent_only"):
        carrier_records = [
            record
            for record in successful
            if record.context_mode == carrier and record.condition == "own"
        ]
        if carrier_records:
            carrier_comparison[carrier] = {
                "source_follow_accuracy": mean(
                    record.content_follow_correct for record in carrier_records
                ),
                "source_probability_mean": mean(
                    record.source_probability for record in carrier_records
                ),
            }
    if drop_records:
        carrier_comparison["drop"] = {
            "source_follow_accuracy": mean(
                record.target_retained for record in drop_records
            ),
            "source_probability_mean": mean(
                record.target_probability for record in drop_records
            ),
        }
    position_matched_drop_records = [
        record for record in successful if record.condition == "drop_position_matched"
    ]
    if position_matched_drop_records:
        carrier_comparison["drop_position_matched"] = {
            "source_follow_accuracy": mean(
                record.target_retained for record in position_matched_drop_records
            ),
            "source_probability_mean": mean(
                record.target_probability for record in position_matched_drop_records
            ),
        }
    carrier_deltas: dict[str, float] = {}
    carrier_probability = {
        carrier: values["source_probability_mean"]
        for carrier, values in carrier_comparison.items()
    }
    for left, right in (
        ("full", "drop"),
        ("prompt_only", "drop"),
        ("latent_only", "drop"),
        ("full", "prompt_only"),
        ("full", "latent_only"),
    ):
        if left in carrier_probability and right in carrier_probability:
            carrier_deltas[f"{left}_minus_{right}"] = (
                carrier_probability[left] - carrier_probability[right]
            )
    lengths: dict[str, dict[str, float]] = {}
    sizes: dict[str, dict[str, float]] = {}
    for mode in (
        "full",
        "prompt_only",
        "latent_only",
    ):
        group = [r for r in successful if r.context_mode == mode and r.cache_present]
        if group:
            seqs = [r.cache_sequence_length for r in group]
            byte_values = [r.cache_bytes for r in group]
            lengths[mode] = {"mean": mean(seqs), "min": min(seqs), "max": max(seqs)}
            sizes[mode] = {"mean": mean(byte_values), "max": max(byte_values)}
    return ReceiverAcquisitionMetrics(
        n_samples=n_samples,
        n_records_expected=expected,
        n_records_successful=len(successful),
        n_errors=len(records) - len(successful),
        cells=cells,
        source_probability_delta_vs_drop=deltas,
        cross_source_follow_accuracy=cross_follow,
        cross_target_retention_accuracy=cross_retention,
        drop_target_match_accuracy=mean(r.target_retained for r in drop_records)
        if drop_records
        else 0.0,
        drop_predicted_digit_counts={
            digit: sum(r.predicted_digit == int(digit) for r in drop_records)
            for digit in CANDIDATE_DIGITS
        },
        runtime_total_sec=runtime,
        runtime_per_sample_sec=runtime / n_samples if n_samples else 0.0,
        runtime_context_build_sec=context_runtime,
        runtime_by_cell_sec={
            cell: sum(
                r.latency_sec or 0.0
                for r in records
                if (
                    f"{r.context_mode}/{r.condition}"
                    if r.condition != "drop"
                    else "drop"
                )
                == cell
            )
            for cell in cells
        },
        cache_sequence_length_stats=lengths,
        cache_bytes_stats=sizes,
        carrier_comparison=carrier_comparison,
        carrier_probability_deltas=carrier_deltas,
    )
