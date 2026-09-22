from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SampleInterventionRecord:
    """Structured sample-level result for latent intervention experiments."""

    sample_id: str | int
    source_sample_id: str | int | None
    condition: str  # "own", "cross", "drop", "zero"
    question: str
    gold: str | None
    prediction: str | None
    raw_prediction: str
    correct: bool
    model: str
    task: str
    latent_steps: int
    seed: int
    sample_index: int = 0
    sample_key: str = ""
    source_sample_index: int | None = None
    source_sample_key: str | None = None
    target_cache_seq_len: int = 0
    source_cache_seq_len: int = 0
    cache_seq_len_delta: int = 0
    cache_present: bool = False
    num_layers: int = 0
    cache_dtype: str | None = None
    latency: float | None = None
    generated_tokens: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InterventionMetrics:
    """Aggregate metrics across intervention conditions."""

    accuracy_own: float = 0.0
    accuracy_cross: float = 0.0
    accuracy_drop: float = 0.0
    accuracy_zero: float = 0.0
    accuracy_by_condition: dict[str, float] = field(default_factory=dict)
    n_correct_by_condition: dict[str, int] = field(default_factory=dict)
    answer_change_rate_cross: float = 0.0
    answer_change_rate_drop: float = 0.0
    answer_change_rate_zero: float = 0.0
    answer_change_rate_by_condition: dict[str, float] = field(default_factory=dict)
    accuracy_delta_own_cross: float = 0.0
    accuracy_delta_own_drop: float = 0.0
    accuracy_delta_own_zero: float = 0.0
    n_samples: int = 0
    n_own_correct: int = 0
    n_cross_correct: int = 0
    n_drop_correct: int = 0
    n_zero_correct: int = 0
    runtime_total: float = 0.0
    runtime_per_sample: float = 0.0
    runtime_own: float = 0.0
    runtime_cross: float = 0.0
    runtime_drop: float = 0.0
    runtime_zero: float = 0.0
    runtime_by_condition: dict[str, float] = field(default_factory=dict)
    paired_transitions: dict[str, Any] = field(default_factory=dict)
    harm_rescue: dict[str, dict[str, int]] = field(default_factory=dict)
    correctness_patterns: dict[str, int] = field(default_factory=dict)
    mcnemar_tests: dict[str, Any] = field(default_factory=dict)
    cache_seq_len_stats: dict[str, Any] = field(default_factory=dict)
    ram_usage_stats: dict[str, Any] = field(default_factory=dict)
    max_new_tokens: int = 0
    tokens_generated_mean: dict[str, float] = field(default_factory=dict)
    tokens_generated_max: dict[str, int] = field(default_factory=dict)
    truncation_rate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_mlflow_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {
            "n_samples": float(self.n_samples),
            "runtime_total": round(self.runtime_total, 4),
            "runtime_per_sample": round(self.runtime_per_sample, 4),
        }

        # Dynamic condition metrics
        for cond, acc in self.accuracy_by_condition.items():
            metrics[f"accuracy/{cond}"] = float(acc)
        for cond, n_corr in self.n_correct_by_condition.items():
            metrics[f"n_{cond}_correct"] = float(n_corr)
        for cond, rt in self.runtime_by_condition.items():
            metrics[f"runtime/{cond}"] = round(float(rt), 4)
        for cond, cr in self.answer_change_rate_by_condition.items():
            metrics[f"answer_change_rate/{cond}"] = float(cr)

        # Standard condition metrics
        if self.accuracy_own or "own" in self.accuracy_by_condition:
            metrics["accuracy/own"] = self.accuracy_own
            metrics["n_own_correct"] = float(self.n_own_correct)
            metrics["runtime/own"] = round(self.runtime_own, 4)
        if self.accuracy_cross or "cross" in self.accuracy_by_condition:
            metrics["accuracy/cross"] = self.accuracy_cross
            metrics["n_cross_correct"] = float(self.n_cross_correct)
            metrics["answer_change_rate/cross"] = self.answer_change_rate_cross
            metrics["accuracy_delta/own_cross"] = self.accuracy_delta_own_cross
            metrics["runtime/cross"] = round(self.runtime_cross, 4)
        if self.accuracy_drop or "drop" in self.accuracy_by_condition:
            metrics["accuracy/drop"] = self.accuracy_drop
            metrics["n_drop_correct"] = float(self.n_drop_correct)
            metrics["answer_change_rate/drop"] = self.answer_change_rate_drop
            metrics["accuracy_delta/own_drop"] = self.accuracy_delta_own_drop
            metrics["runtime/drop"] = round(self.runtime_drop, 4)
        if self.accuracy_zero or "zero" in self.accuracy_by_condition:
            metrics["accuracy/zero"] = self.accuracy_zero
            metrics["n_zero_correct"] = float(self.n_zero_correct)
            metrics["answer_change_rate/zero"] = self.answer_change_rate_zero
            metrics["accuracy_delta/own_zero"] = self.accuracy_delta_own_zero
            metrics["runtime/zero"] = round(self.runtime_zero, 4)

        # Harm & rescue metrics
        for cond, hr in self.harm_rescue.items():
            metrics[f"transition/{cond}/harm"] = float(hr.get("harm", 0))
            metrics[f"transition/{cond}/rescue"] = float(hr.get("rescue", 0))

        if self.max_new_tokens > 0:
            metrics["max_new_tokens"] = float(self.max_new_tokens)
        for cond, val in self.tokens_generated_mean.items():
            metrics[f"tokens_mean/{cond}"] = float(val)
        for cond, val in self.tokens_generated_max.items():
            metrics[f"tokens_max/{cond}"] = float(val)
        for cond, val in self.truncation_rate.items():
            metrics[f"truncation_rate/{cond}"] = float(val)

        return metrics
