"""Use case names, MLflow experiment names and storage path suffixes."""

from types import SimpleNamespace

import pytest

from latentmas_reprop.domain.models import use_cases
from latentmas_reprop.infrastructure.paths.naming import build_params_stem

BASE = "ls20_seed42_temp0.6_top0.95_mt2048_float16"


def test_experiment_names_cover_exactly_the_six_use_cases():
    assert set(use_cases.MLFLOW_EXPERIMENT_NAMES) == set(use_cases.USE_CASES)
    assert {use_cases.mlflow_experiment_name(name) for name in use_cases.USE_CASES} == {
        "latentmas/benchmark",
        "latentmas/intervention/kv",
        "latentmas/intervention/continuation",
        "latentmas/representation/acquisition",
        "latentmas/representation/trajectory",
        "latentmas/representation/sender_probe",
    }


def test_unknown_use_case_is_rejected():
    with pytest.raises(ValueError, match="Unknown use case"):
        use_cases.mlflow_experiment_name("receiver_reasoning")


@pytest.mark.parametrize(
    "flag, suffix",
    [
        ("intervention_kv", "intervention_kv"),
        ("representation_acquisition", "representation_acquisition"),
        ("representation_trajectory", "representation_trajectory"),
    ],
)
def test_params_stem_uses_use_case_name_suffix(flag, suffix):
    args = SimpleNamespace(latent_steps=20, dtype="float16", **{flag: True})
    stem = build_params_stem(args)
    assert stem.startswith(BASE)
    assert f"_{suffix}" in stem


def test_params_stem_without_use_case_flag_is_benchmark_stem():
    assert build_params_stem(SimpleNamespace(latent_steps=20, dtype="float16")) == BASE
