import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class InterventionCondition(StrEnum):
    """Intervention conditions for latent communication ablation."""

    OWN = "own"
    CROSS = "cross"
    DROP = "drop"
    ZERO = "zero"


@dataclass
class Agent:
    """Agent entity participating in multi-agent reasoning."""

    name: str
    role: str


def default_agents() -> list[Agent]:
    """Default four-agent pipeline: Planner, Critic, Refiner, Judger."""
    return [
        Agent(name="Planner", role="planner"),
        Agent(name="Critic", role="critic"),
        Agent(name="Refiner", role="refiner"),
        Agent(name="Judger", role="judger"),
    ]


@dataclass
class ProblemSample:
    """Benchmark problem sample."""

    question: str
    solution: str
    gold: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "solution": self.solution,
            "gold": self.gold,
            **self.extra,
        }


@dataclass
class AgentTrace:
    """Execution trace of an agent's step."""

    name: str
    role: str
    input: str
    output: str
    input_ids: list[int] | None = None
    input_tokens: list[str] | None = None
    latent_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "input": self.input,
            "output": self.output,
        }
        if self.input_ids is not None:
            d["input_ids"] = self.input_ids
        if self.input_tokens is not None:
            d["input_tokens"] = self.input_tokens
        if self.latent_steps is not None:
            d["latent_steps"] = self.latent_steps
        return d


@dataclass
class EvaluationResult:
    """Evaluation result for a single sample."""

    question: str
    gold: str | None
    solution: str
    prediction: str | None
    raw_prediction: str
    agents: list[dict[str, Any]]
    correct: bool
    context: str | None = None
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        res = {
            "question": self.question,
            "gold": self.gold,
            "solution": self.solution,
            "prediction": self.prediction,
            "raw_prediction": self.raw_prediction,
            "agents": self.agents,
            "correct": self.correct,
        }
        if self.context is not None:
            res["context"] = self.context
        if self.error_msg is not None:
            res["error_msg"] = self.error_msg
        return res


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark run metrics."""

    method: str
    model: str
    split: str
    seed: int
    max_samples: int
    accuracy: float
    correct: int
    total_time_sec: float
    time_per_sample_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_sample_key(task: str, split: str, question: str) -> str:
    """Generate deterministic sha256 identifier for a dataset sample."""
    norm_q = " ".join(question.strip().split())
    raw = f"{task}:{split}:{norm_q}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


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
    method_note: str = (
        "latent-only retains tail KV positions without rebasing their rotary positions"
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
        return result
