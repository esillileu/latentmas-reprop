from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReceiverAcquisitionRecord:
    """One receiver next-token observation under a sender-cache condition."""

    sample_id: str
    sample_index: int
    sample_key: str
    target_digit: int
    source_sample_id: str | None
    source_sample_index: int | None
    source_sample_key: str | None
    source_digit: int | None
    context_mode: str
    condition: str
    candidate_probabilities: dict[str, float]
    candidate_log_probabilities: dict[str, float]
    top_token_candidates: list[dict[str, Any]]
    candidate_mass: float
    predicted_digit: int | None
    source_probability: float | None
    source_log_probability: float | None
    source_rank: int | None
    target_probability: float | None
    target_log_probability: float | None
    target_rank: int | None
    source_margin: float | None
    content_follow_correct: bool | None
    target_retained: bool | None
    cache_present: bool
    cache_sequence_length: int
    cache_bytes: int
    num_layers: int
    cache_dtype: str | None
    original_full_seq_len: int
    retained_tail_len: int
    retained_tail_start_position: int | None
    retained_original_position_end: int | None
    receiver_position_start: int
    receiver_prompt_tokens: int
    latency_sec: float | None
    error: str | None
    hidden_state_artifact_key: str | None
    seed: int
    model: str
    task: str
    latent_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReceiverAcquisitionMetrics:
    """Nested receiver-acquisition summary plus stable MLflow flattening."""

    n_samples: int
    n_records_expected: int
    n_records_successful: int
    n_errors: int
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_probability_delta_vs_drop: dict[str, float] = field(default_factory=dict)
    cross_source_follow_accuracy: dict[str, float] = field(default_factory=dict)
    cross_target_retention_accuracy: dict[str, float] = field(default_factory=dict)
    drop_target_match_accuracy: float = 0.0
    drop_predicted_digit_counts: dict[str, int] = field(default_factory=dict)
    runtime_total_sec: float = 0.0
    runtime_per_sample_sec: float = 0.0
    runtime_context_build_sec: float = 0.0
    runtime_by_cell_sec: dict[str, float] = field(default_factory=dict)
    cache_sequence_length_stats: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    cache_bytes_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    carrier_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    carrier_probability_deltas: dict[str, float] = field(default_factory=dict)
    probe_metrics: dict[str, float] = field(default_factory=dict)
    research_matrix: dict[str, Any] = field(default_factory=dict)
    method_note: str = (
        "carrier KV rotations are retained; canonical receiver positions start at "
        "the original full cache length"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mlflow_metrics(self) -> dict[str, float]:
        result: dict[str, float] = {
            "count/samples": float(self.n_samples),
            "count/records_successful": float(self.n_records_successful),
            "count/errors": float(self.n_errors),
            "runtime/total_sec": self.runtime_total_sec,
            "runtime/per_sample_sec": self.runtime_per_sample_sec,
            "runtime/context_build_sec": self.runtime_context_build_sec,
            "accuracy/drop/target_match": self.drop_target_match_accuracy,
        }
        for cell, values in self.cells.items():
            mode, condition = cell.split("/", 1) if "/" in cell else (cell, "")
            metric_cell = f"{mode}/{condition}" if condition else mode
            if values.get("content_follow_accuracy") is not None:
                result[f"accuracy/{metric_cell}/source_follow"] = float(
                    values["content_follow_accuracy"]
                )
            for source, suffix in (
                ("mean_source_probability", "prob"),
                ("mean_source_rank", "rank"),
                ("mean_source_margin", "margin"),
            ):
                if values.get(source) is not None:
                    result[f"{suffix}/{metric_cell}/source_mean"] = float(
                        values[source]
                    )
            if values.get("mean_target_probability") is not None:
                result[f"prob/{metric_cell}/target_mean"] = float(
                    values["mean_target_probability"]
                )
            if values.get("mean_candidate_mass") is not None:
                result[f"prob/{metric_cell}/candidate_mass_mean"] = float(
                    values["mean_candidate_mass"]
                )
            result[f"count/{metric_cell}/successful"] = float(
                values.get("n_successful", 0)
            )
        for cell, value in self.source_probability_delta_vs_drop.items():
            result[f"prob_delta/{cell}_vs_drop"] = value
        for mode, value in self.cross_target_retention_accuracy.items():
            result[f"accuracy/{mode}/cross/target_retention"] = value
        for digit, count in self.drop_predicted_digit_counts.items():
            result[f"distribution/drop/predicted_digit/{digit}"] = float(count)
        for cell, value in self.runtime_by_cell_sec.items():
            result[f"runtime/{cell}_sec"] = value
        for mode, stats in self.cache_sequence_length_stats.items():
            for name, value in stats.items():
                result[f"cache/{mode}/sequence_length_{name}"] = value
        for mode, stats in self.cache_bytes_stats.items():
            for name, value in stats.items():
                result[f"cache/{mode}/bytes_{name}"] = value
        for carrier, values in self.carrier_comparison.items():
            for name, value in values.items():
                result[f"carrier/{carrier}/{name}"] = value
                result[f"{carrier}/{name}"] = value
        for comparison, value in self.carrier_probability_deltas.items():
            result[f"carrier_delta/{comparison}"] = value
        result.update(self.probe_metrics)
        return result
