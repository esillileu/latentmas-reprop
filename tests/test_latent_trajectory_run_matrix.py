"""Alignment lists expand into scalar runs without sweeping sample/condition lists."""

import json
from pathlib import Path

import pytest

from src.run import latent_trajectory as cli

PRESET = "latent_trajectory/gsm8k_q314"


def test_alignment_run_plan_and_cli_override(capsys):
    cli.main(["-c", PRESET, "--dry-run", "--sample_indices", "37", "0"])
    plans = json.loads(capsys.readouterr().out)["runs"]
    assert [p["config"]["latent_space_realign"] for p in plans] == [False, True]
    assert len({p["directory"] for p in plans}) == 2
    assert all(
        p["conditions"] == list("abcd") and p["sample_indices"] == [0, 37]
        for p in plans
    )
    for flag, expected in (
        ("--latent_space_realign", True),
        ("--no_latent_space_realign", False),
    ):
        cli.main(["-c", PRESET, "--dry-run", flag])
        plan = json.loads(capsys.readouterr().out)
        assert "runs" not in plan
        assert plan["config"]["latent_space_realign"] is expected
        assert plan["directory"] == plans[int(expected)]["directory"]


def test_alignment_runs_execute_sequentially(monkeypatch):
    collected = []
    monkeypatch.setattr(
        cli,
        "run_collection",
        lambda args, config, directory: collected.append(
            (
                args.latent_space_realign,
                config["latent_space_realign"],
                args.sample_indices,
                directory,
            )
        ),
    )
    cli.main(["-c", PRESET, "--sample_indices", "12", "0"])
    assert [(a, c, samples) for a, c, samples, _ in collected] == [
        (False, False, [0, 12]),
        (True, True, [0, 12]),
    ]
    assert collected[0][3] != collected[1][3]


@pytest.mark.parametrize("value", ["[]", "[false, invalid]"])
def test_alignment_sweep_rejects_invalid_values(tmp_path, value):
    source = Path(f"configs/{PRESET}.yaml").read_text()
    preset = tmp_path / "alignment.yaml"
    preset.write_text(source.replace("[false, true]", value))
    with pytest.raises(SystemExit) as error:
        cli.main(["-c", str(preset.with_suffix("")), "--dry-run"])
    assert error.value.code == 2


@pytest.mark.parametrize("task,indices", [("gsm8k", [0, 1288]), ("medqa", [0, 145])])
def test_default_14b_presets(task, indices, capsys):
    cli.main(["-c", f"latent_trajectory/{task}_q314", "--dry-run"])
    plans = json.loads(capsys.readouterr().out)["runs"]
    assert [p["config"]["latent_space_realign"] for p in plans] == [False, True]
    for plan in plans:
        assert plan["config"]["task"] == task
        assert plan["config"]["model_name"] == "Qwen/Qwen3-14B"
        assert plan["sample_indices"] == indices
        assert [r["sample_index"] for r in plan["mlflow_runs"]] == indices
        assert len({r["run_name"] for r in plan["mlflow_runs"]}) == 2
        assert plan["max_samples"] == 2
        assert plan["conditions"] == list("abcd")
        assert plan["config"]["latent_steps"] == 80
    if task == "medqa":
        from types import SimpleNamespace

        from latentmas_reprop.domain.services.prompts import build_initial_messages
        from latentmas_reprop.infrastructure.evaluators.evaluator import (
            DEFAULT_EVALUATOR,
        )

        messages = build_initial_messages(
            "Example multiple-choice question", SimpleNamespace(**plans[0]["config"])
        )
        assert "selected from A,B,C,D" in messages[-1]["content"]
        assert DEFAULT_EVALUATOR.evaluate("medqa", r"Answer: \boxed{A}", "a") == (
            "a",
            True,
            None,
        )
