"""Tests for the one-time use case rename migration tool."""

import json
from pathlib import Path

import pytest

from tools.migrations.use_case_names import migrate

BASE = "ls20_seed42_temp0.6_top0.95_mt2048_float16"


@pytest.mark.parametrize(
    "old, new",
    [
        (f"{BASE}_single_intervention", f"{BASE}_intervention_continuation"),
        (
            f"{BASE}_single_intervention_sig_Answer_role_judger",
            f"{BASE}_intervention_continuation_sig_Answer_role_judger",
        ),
        (f"{BASE}_intervention_cross_shift_1", f"{BASE}_intervention_kv_cross_shift_1"),
        (
            f"{BASE}_acquisition_modes_full-latent_only_cond_cross-own",
            f"{BASE}_representation_acquisition_modes_full-latent_only_cond_cross-own",
        ),
        (f"{BASE}_think_realign_intervention", f"{BASE}_think_realign_intervention_kv"),
        (f"{BASE}_trajectory_smoke", f"{BASE}_representation_trajectory_smoke"),
    ],
)
def test_params_stem_rename(old, new):
    if old.endswith("realign_intervention"):
        new = f"{BASE}_think_realign_intervention_kv"
    assert migrate.new_params_stem(old) == new


@pytest.mark.parametrize(
    "stem",
    [
        BASE,
        f"{BASE}_think_realign",
        f"{BASE}_intervention_kv",
        f"{BASE}_sig_intervention",
        "ls20_seed42_steps1-20_bfloat16",
    ],
)
def test_unaffected_stems_are_left_alone(stem):
    assert migrate.new_params_stem(stem) is None


def make_condition(root: Path, content: str) -> None:
    condition = root / "samples" / "000000" / "ple"
    condition.mkdir(parents=True)
    (condition / "kv.pt").write_bytes(content.encode())
    (condition / "result.json").write_text(json.dumps({"files": ["kv.pt"]}))


def layout(tmp_path: Path, stem: str) -> Path:
    return tmp_path / "storage" / "experiments" / "gsm8k" / "Qwen3-8B" / "latent_mas" / "sequential" / stem


def run(tmp_path: Path, *extra: str) -> int:
    return migrate.main(
        [
            "--experiments-dir", str(tmp_path / "storage" / "experiments"),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--cache-dir", str(tmp_path / ".cache"),
            "--skip-mlflow",
            *extra,
        ]
    )  # fmt: skip


def test_dry_run_changes_nothing(tmp_path):
    source = layout(tmp_path, f"{BASE}_single_intervention")
    make_condition(source, "tensor")
    assert run(tmp_path) == 0
    assert source.is_dir()
    assert not layout(tmp_path, f"{BASE}_intervention_continuation").exists()


def test_execute_moves_verified_results_and_keeps_them_complete(tmp_path):
    source = layout(tmp_path, f"{BASE}_single_intervention")
    make_condition(source, "tensor")
    assert run(tmp_path, "--execute") == 0
    dest = layout(tmp_path, f"{BASE}_intervention_continuation")
    assert not source.exists()
    assert (dest / "samples" / "000000" / "ple" / "kv.pt").read_bytes() == b"tensor"
    from latentmas_reprop.infrastructure.storage.tensor_store import (
        is_condition_complete,
    )

    assert is_condition_complete(dest / "samples" / "000000" / "ple")
    reports = list((tmp_path / "storage" / "migrations").glob("use_case_names_*.json"))
    assert len(reports) == 1
    # Re-running is a no-op.
    assert run(tmp_path, "--execute") == 0


def test_destination_conflict_blocks_the_move(tmp_path):
    source = layout(tmp_path, f"{BASE}_single_intervention")
    make_condition(source, "old")
    dest = layout(tmp_path, f"{BASE}_intervention_continuation")
    make_condition(dest, "new")
    assert run(tmp_path, "--execute") == 1
    assert (source / "samples" / "000000" / "ple" / "kv.pt").read_bytes() == b"old"
    assert (dest / "samples" / "000000" / "ple" / "kv.pt").read_bytes() == b"new"


def test_local_analysis_directories_are_renamed(tmp_path):
    (tmp_path / "artifacts" / "receiver_acquisition").mkdir(parents=True)
    (tmp_path / "artifacts" / "receiver_acquisition" / "a.csv").write_text("x")
    cache = tmp_path / ".cache" / "evaluation" / "receiver_acquisition"
    cache.mkdir(parents=True)
    assert run(tmp_path, "--execute") == 0
    assert (tmp_path / "artifacts" / "representation_acquisition" / "a.csv").is_file()
    assert (tmp_path / ".cache" / "evaluation" / "representation_acquisition").is_dir()


def test_mlflow_references_and_experiment_are_realigned(tmp_path):
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=uri)
    old_rel = f"gsm8k/Qwen3-8B/latent_mas/sequential/{BASE}_single_intervention"
    experiment_id = client.create_experiment("latentmas_single_intervention")
    run_info = client.create_run(
        experiment_id,
        run_name=old_rel,
        tags={
            "rel_dir": old_rel,
            "run_name": old_rel,
            "data_dir": f"/data/storage/experiments/{old_rel}",
        },
    ).info
    client.set_terminated(run_info.run_id)
    mlflow.set_tracking_uri(uri)
    experiments_dir = tmp_path / "storage" / "experiments"

    summary = migrate.run_mlflow(
        uri, experiments_dir, execute=True, rename_experiments=True
    )

    new_rel = f"gsm8k/Qwen3-8B/latent_mas/sequential/{BASE}_intervention_continuation"
    run = client.get_run(run_info.run_id)
    assert summary["runs_updated"] == 1
    assert run.data.tags["rel_dir"] == new_rel
    assert run.data.tags["run_name"] == new_rel
    assert run.data.tags["data_dir"] == f"/data/storage/experiments/{new_rel}"
    assert run.info.run_name == new_rel
    experiment = client.get_experiment(experiment_id)
    assert experiment.name == "latentmas/intervention/continuation"
    assert client.get_run(run_info.run_id).info.experiment_id == experiment_id
