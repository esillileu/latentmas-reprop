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
