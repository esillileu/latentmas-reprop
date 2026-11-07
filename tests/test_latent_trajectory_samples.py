"""Explicit sample selection preserves dataset order, cache paths and run identity."""

import json
from types import SimpleNamespace

import pytest

from latentmas_reprop.application.latent_trajectory_samples import (
    select_samples,
)
from src.run.latent_trajectory import main


def test_default_and_explicit_selection():
    dataset = list("abcd")
    args = SimpleNamespace(max_samples=1, sample_indices=None)
    assert list(select_samples(iter(dataset), args)) == [(0, "a")]
    args.sample_indices = [3, 1, 3]
    assert list(select_samples(iter(dataset), args)) == [(1, "b"), (3, "d")]
    args.sample_indices = [1, 8]
    with pytest.raises(ValueError, match=r"outside the dataset.*8"):
        select_samples(iter(dataset), args)
    args.sample_indices = None
    args.max_samples = -1
    assert list(select_samples(iter(dataset), args)) == list(enumerate(dataset))


def test_cli_selection_keeps_existing_run_identity(capsys):
    main(["--dry-run"])
    default = json.loads(capsys.readouterr().out)
    main(["--dry-run", "--sample_indices", "37", "0", "12", "37"])
    selected = json.loads(capsys.readouterr().out)
    assert default["directory"] == selected["directory"]
    assert default["sample_indices"] is None
    assert selected["sample_indices"] == [0, 12, 37]
    assert selected["max_samples"] == 1
    with pytest.raises(SystemExit) as error:
        main(["--dry-run", "--sample_indices", "-1"])
    assert error.value.code == 2


def test_cli_alignment_overrides_preset_and_separates_caches(tmp_path, capsys):
    from pathlib import Path

    main(["--dry-run"])
    default = json.loads(capsys.readouterr().out)
    main(["--dry-run", "--latent_space_realign"])
    enabled = json.loads(capsys.readouterr().out)
    main(["--dry-run", "--no_latent_space_realign"])
    disabled = json.loads(capsys.readouterr().out)
    assert enabled["config"]["latent_space_realign"] is True
    assert disabled["config"]["latent_space_realign"] is False
    assert default["directory"] == disabled["directory"]
    assert enabled["directory"] != disabled["directory"]
    preset = tmp_path / "aligned.yaml"
    preset.write_text(
        Path("configs/latent_trajectory/gsm8k_q38.yaml")
        .read_text()
        .replace("latent_space_realign: false", "latent_space_realign: true")
    )
    main(["-c", str(preset.with_suffix("")), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["config"]["latent_space_realign"] is True
    main(["-c", str(preset.with_suffix("")), "--dry-run", "--no_latent_space_realign"])
    assert json.loads(capsys.readouterr().out)["directory"] == disabled["directory"]
    with pytest.raises(SystemExit) as error:
        main(["--dry-run", "--latent_space_realign", "--no_latent_space_realign"])
    assert error.value.code == 2


def test_id_selection_preserves_indices_and_rejects_missing_ids():
    dataset = [{"id": value} for value in [0, 239, 127]]
    args = SimpleNamespace(max_samples=1, sample_ids=["127", "239"])
    assert select_samples(iter(dataset), args) == [(1, dataset[1]), (2, dataset[2])]
    args.sample_ids = ["missing"]
    with pytest.raises(ValueError, match="Sample IDs outside the dataset"):
        select_samples(iter(dataset), args)


@pytest.mark.parametrize(
    "task,ids",
    [
        ("arc_challenge", ["MCAS_2006_9_1", "AKDE&ED_2008_8_3"]),
        ("mbppplus", ["127", "239"]),
    ],
)
def test_id_presets_and_index_override(task, ids, capsys):
    main(["-c", f"latent_trajectory/{task}_q314", "--dry-run"])
    plans = json.loads(capsys.readouterr().out)["runs"]
    assert [p["config"]["latent_space_realign"] for p in plans] == [False, True]
    assert all(p["sample_ids"] == ids and p["mlflow_runs"] is None for p in plans)
    main(["-c", f"latent_trajectory/{task}_q314", "--sample_indices", "0", "--dry-run"])
    overridden = json.loads(capsys.readouterr().out)["runs"]
    assert all(
        p["sample_ids"] is None and p["sample_indices"] == [0] for p in overridden
    )


@pytest.mark.parametrize("task", ["arc_challenge", "mbppplus"])
def test_loaders_preserve_original_ids(monkeypatch, task):
    from latentmas_reprop.infrastructure.datasets import loaders

    record = {
        "id": "MCAS_2006_9_1",
        "task_id": 127,
        "question": "Question",
        "choices": {"label": ["A", "B"], "text": ["one", "two"]},
        "answerKey": "A",
        "prompt": "Write code",
        "test_list": ["a", "b", "c"],
        "test": "assert True",
    }
    monkeypatch.setattr(loaders, "load_dataset", lambda *a, **kw: [record])
    item = next(getattr(loaders, f"load_{task}")())
    assert item["id"] == (record["id"] if task == "arc_challenge" else 127)
