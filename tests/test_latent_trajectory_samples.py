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
