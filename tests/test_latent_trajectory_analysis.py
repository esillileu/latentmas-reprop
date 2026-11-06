"""Whitespace-only token preprocessing and strict adjacency of saved readouts."""

import json

import pytest

from latentmas_reprop.application.latent_trajectory_analysis import (
    adjacent_readout_metrics,
    analyze_readouts,
    readout_text,
)


class CharacterTokenizer:
    def decode(self, ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        return "".join(map(chr, ids))

    def encode(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return list(map(ord, text))


def record(text, condition="a", step=1):
    return {
        "condition": condition,
        "step": step,
        "generated_token_ids": [*map(ord, text), 0],
        "content_length": len(text),
        "generated_text": text,
        "correct": True,
        "prediction": "1",
        "budget_exhausted": False,
    }


def save(sample, text, condition, step, **updates):
    path = sample / "continuations" / condition / f"step_{step:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**record(text, condition, step), **updates}))


def test_whitespace_only_preserves_symbols_tags_and_unicode():
    text = " \n<think>\t -1 + 2 = 1\u00a0\n</think> e\u0301  "
    assert (
        readout_text(record(text), CharacterTokenizer())
        == "<think> -1 + 2 = 1 </think> e\u0301"
    )
    assert readout_text(record(" \n\t"), CharacterTokenizer()) == ""
    with pytest.raises(ValueError, match="content_length"):
        readout_text({**record("x"), "content_length": -1}, CharacterTokenizer())


def test_adjacent_distances_missing_steps_and_empty_sequences(tmp_path):
    sample = tmp_path / "run/samples/000660"
    save(sample, "one\n two", "a", 1)
    save(sample, " one  two ", "a", 2)
    save(sample, "different", "a", 4)
    save(sample, "", "b", 1)
    save(sample, "x", "b", 2, correct=False, budget_exhausted=True)
    save(sample, "", "c", 1)
    save(sample, "", "c", 2)
    save(sample, "x", "d", 1)
    save(sample, "x", "d", 2, status="running")
    metadata = {"latent_step_start": 1, "latent_steps": 4}
    output = tmp_path / "output"
    result = analyze_readouts(
        sample.parents[1], metadata, CharacterTokenizer(), output
    )[0]
    assert [(r["condition"], r["step"]) for r in result["rows"]] == [
        ("a", 2),
        ("b", 2),
        ("c", 2),
    ]
    assert [r["normalized_distance"] for r in result["rows"]] == [0, 1, 0]
    assert result["rows"][1]["previous_correct"] is True
    assert result["rows"][1]["correct"] is False
    assert result["rows"][1]["budget_exhausted"] is True
    assert result["missing_steps"]["a"] == [3]
    assert result["missing_steps"]["d"] == [2, 3, 4]
    assert (output / "000660/token_distances.json").exists()
    assert (output / "source.json").exists()
    save(sample, "x", "a", 3, step_unused=True)
    path = sample / "continuations/a/step_003.json"
    data = json.loads(path.read_text())
    data["step"] = 2
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        adjacent_readout_metrics(sample, metadata, CharacterTokenizer())


def test_cli_reports_and_tracks_source_separately(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from src.run import latent_trajectory_analysis as cli

    directory = tmp_path / "gsm8k/TestModel/run"
    sample = directory / "samples/000000"
    save(sample, "one\n two", "a", 1)
    save(sample, "one two!", "a", 2)
    metadata = {
        "model_name": "TestModel",
        "task": "gsm8k",
        "latent_space_realign": False,
        "latent_step_start": 1,
        "latent_steps": 2,
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(
        cli, "load_trajectory_tokenizer", lambda _: CharacterTokenizer()
    )
    tracker = Mock()
    log = Mock()
    trajectories = Mock()
    monkeypatch.setattr(cli, "MLflowTracker", lambda: tracker)
    monkeypatch.setattr(cli, "log_analysis_results_to_mlflow", log)
    monkeypatch.setattr(cli, "analyze_trajectories", trajectories)
    output = tmp_path / "analysis"
    cli.main([str(directory), "--output-dir", str(output), "--continuations-only"])
    trajectories.assert_not_called()
    assert log.call_args.args[:2] == (tracker, "latent_trajectory/gsm8k/TestModel/run")
    assert log.call_args.kwargs["metrics"] == {"pairs": 1}
    assert (output / "000000/token_distances.csv").exists()
    assert (output / "000000/adjacent_distances.png").exists()
    assert "1 → 2" in (output / "summary.md").read_text()
    cli.main([str(directory), "--output-dir", str(output)])
    trajectories.assert_called_once_with(directory.resolve())


def test_tokenizer_pinned_local_and_identity_checked(monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from unittest.mock import Mock

    from latentmas_reprop.infrastructure.models import trajectory_tokenizer as adapter

    tokenizer = SimpleNamespace(
        init_kwargs={},
        backend_tokenizer=SimpleNamespace(to_str=lambda: "tokenizer-data"),
    )
    loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(adapter.AutoTokenizer, "from_pretrained", loader)
    metadata = {
        "tokenizer": "TestModel",
        "tokenizer_revision": None,
        "model_revision": "saved-commit",
        "state_identity": {"tokenizer": hashlib.sha256(b"tokenizer-data").hexdigest()},
    }
    assert adapter.load_trajectory_tokenizer(metadata) is tokenizer
    loader.assert_called_once_with(
        "TestModel", revision="saved-commit", local_files_only=True
    )
    metadata["state_identity"]["tokenizer"] = "different"
    with pytest.raises(ValueError, match="differs"):
        adapter.load_trajectory_tokenizer(metadata)


def test_boxed_only_b_readouts_excluded_without_bridging(tmp_path):
    sample = tmp_path / "samples/000000"
    texts = [
        "reasoning one",
        r" $\boxed{24}$ ",
        r"\boxed{18}",
        "reasoning four",
        "reasoning five",
    ]
    for condition in "ab":
        for step, text in enumerate(texts, 1):
            save(sample, text, condition, step, correct=(step != 2))
    # Short outputs containing reasoning or thinking tags are retained.
    save(sample, r"18, so \boxed{18}", "b", 6)
    save(sample, r"</think>\boxed{18}", "b", 7)
    result = adjacent_readout_metrics(
        sample, {"latent_step_start": 1, "latent_steps": 7}, CharacterTokenizer()
    )
    assert result["excluded_steps"] == {"a": [], "b": [2, 3], "c": [], "d": []}
    assert result["missing_steps"]["b"] == []
    b_rows = [r for r in result["rows"] if r["condition"] == "b"]
    assert [(r["previous_step"], r["step"]) for r in b_rows] == [(4, 5), (5, 6), (6, 7)]
    assert len([r for r in result["rows"] if r["condition"] == "a"]) == 4
