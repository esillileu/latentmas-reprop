"""Canonical use case names and MLflow experiment names.

This module is the single place where use case identifiers and the MLflow
experiment each one logs to are defined. Other layers must reference these
constants instead of hard-coding names.
"""

BENCHMARK = "benchmark"
INTERVENTION_KV = "intervention_kv"
INTERVENTION_CONTINUATION = "intervention_continuation"
REPRESENTATION_ACQUISITION = "representation_acquisition"
REPRESENTATION_TRAJECTORY = "representation_trajectory"
REPRESENTATION_SENDER_PROBE = "representation_sender_probe"

USE_CASES = (
    BENCHMARK,
    INTERVENTION_KV,
    INTERVENTION_CONTINUATION,
    REPRESENTATION_ACQUISITION,
    REPRESENTATION_TRAJECTORY,
    REPRESENTATION_SENDER_PROBE,
)

MLFLOW_EXPERIMENT_NAMES = {
    BENCHMARK: "latentmas/benchmark",
    INTERVENTION_KV: "latentmas/intervention/kv",
    INTERVENTION_CONTINUATION: "latentmas/intervention/continuation",
    REPRESENTATION_ACQUISITION: "latentmas/representation/acquisition",
    REPRESENTATION_TRAJECTORY: "latentmas/representation/trajectory",
    REPRESENTATION_SENDER_PROBE: "latentmas/representation/sender_probe",
}


def mlflow_experiment_name(use_case: str) -> str:
    """Return the MLflow experiment name for a use case."""
    try:
        return MLFLOW_EXPERIMENT_NAMES[use_case]
    except KeyError:
        raise ValueError(f"Unknown use case: {use_case!r}") from None
