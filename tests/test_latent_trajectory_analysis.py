"""Partial reports preserve missingness, pairing and generated/prefilled boundaries."""

import json

import pytest

from latentmas_reprop.application.latent_readout_analysis import (
    compare_conditions,
    load_readouts,
    summarize_readouts,
)
from latentmas_reprop.application.latent_readout_features import readout_text
from latentmas_reprop.application.latent_readout_review import review_cases


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


def setup_run(tmp_path, sample_id=1288):
    root = tmp_path / "gsm8k/Model/run"
    sample = root / f"samples/{sample_id:06d}"
    sample.mkdir(parents=True)
    (sample / "sample.json").write_text(
        json.dumps({"index": sample_id, "gold": "18", "thinking_token_ids": [60, 62]})
    )
    metadata = {
        "task": "gsm8k",
        "model_name": "Model",
        "latent_space_realign": False,
        "latent_step_start": 0,
        "latent_steps": 3,
        "split": "test",
    }
    (root / "metadata.json").write_text(json.dumps(metadata))
    return root, sample, metadata


def save(sample, condition, step, text="18", **updates):
    path = sample / "continuations" / condition / f"step_{step:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix("").mkdir(exist_ok=True)
    (path.with_suffix("") / "hidden.pt").touch()
    record = {
        "condition": condition,
        "step": step,
        "status": "complete",
        "rng_provenance": {"seed": 42},
        "identity": {"decoder": "test"},
        "generated_token_ids": [*map(ord, text), 0],
        "generated_length": len(text) + 1,
        "content_length": len(text),
        "correct": True,
        "gold": "18",
        "prediction": "18",
        "finish_reason": "eos",
        "budget_exhausted": False,
        "inserted_prefix_ids": [],
        "inserted_readout_ids": [],
        "generated_text": text,
    }
    path.write_text(json.dumps({**record, **updates}))
    return path


def test_partial_status_no_bridging_or_false_denominators(tmp_path):
    root, sample, meta = setup_run(tmp_path)
    save(sample, "a", 0, correct=False)
    save(sample, "a", 2)
    save(sample, "a", 3, status="running")
    save(sample, "b", 0, text=r"\boxed{18}")
    broken = save(sample, "c", 0)
    broken.write_text("{broken")
    rows, records = load_readouts(root, meta, CharacterTokenizer())
    lookup = {(r["condition"], r["step"]): r for r in rows}
    assert len(rows) == 16
    assert lookup["a", 1]["status"] == "missing"
    assert lookup["a", 3]["status"] == "incomplete"
    assert lookup["c", 0]["status"] == "invalid"
    assert lookup["a", 2]["transition"] is None
    assert lookup["b", 0]["evaluator_correct"] is True
    assert not review_cases(rows, records)
    trajectories, accuracies = summarize_readouts(rows)
    assert trajectories[0]["correct_step_fraction"] == 0.5
    assert (
        next(r for r in accuracies if r["condition"] == "a" and r["step"] == 1)[
            "accuracy"
        ]
        is None
    )
    comparisons = compare_conditions(rows)
    ab = next(
        r
        for r in comparisons
        if r["step"] == 0 and r["left"] == "a" and r["right"] == "b"
    )
    assert ab["n_pairs"] == 1 and ab["n01"] == 1
    assert ab["accuracy_diff"] == -1


def test_boundaries_lengths_and_review_candidates(tmp_path):
    root, sample, meta = setup_run(tmp_path)
    save(sample, "b", 0, text="ab>18", correct=False, inserted_prefix_ids=[60])
    save(sample, "b", 1, text="<c>18", inserted_prefix_ids=[60])
    save(sample, "d", 0, text="18", inserted_prefix_ids=[60], inserted_readout_ids=[62])
    rows, records = load_readouts(root, meta, CharacterTokenizer())
    lookup = {(r["condition"], r["step"]): r for r in rows}
    b = lookup["b", 0]
    assert b["generated_length"] == 6 and b["content_length"] == 5
    assert b["reasoning_token_count"] == 2
    assert b["prefill_think_open_count"] == 1
    assert b["generated_think_open_count"] == 0
    assert lookup["d", 0]["reasoning_token_count"] == 0
    assert lookup["b", 1]["adjacent_token_distance"] is not None
    cases = review_cases(rows, records)
    assert len(cases) == 1 and cases[0]["review_status"] == "unreviewed"
    assert cases[0]["previous_output"] == "ab>18"


def test_invalid_identity_and_missing_hidden_are_visible(tmp_path):
    root, sample, meta = setup_run(tmp_path)
    path = save(sample, "a", 0)
    record = json.loads(path.read_text())
    record["step"] = 1
    path.write_text(json.dumps(record))
    other = save(sample, "b", 0)
    (other.with_suffix("") / "hidden.pt").unlink()
    rows, _ = load_readouts(root, meta)
    assert rows[0]["status"] == "invalid"
    assert rows[4]["status"] == "incomplete"


def test_whitespace_only_preprocessing():
    text = " \n<think>\t -1 + 2 = 1\u00a0\n</think> e\u0301  "
    record = {"generated_token_ids": [*map(ord, text), 0], "content_length": len(text)}
    assert (
        readout_text(record, CharacterTokenizer())
        == "<think> -1 + 2 = 1 </think> e\u0301"
    )
    with pytest.raises(ValueError, match="content_length"):
        readout_text({**record, "content_length": -1}, CharacterTokenizer())


def test_cli_without_tokenizer_or_tensors_still_reports(tmp_path, monkeypatch):
    from src.run import latent_trajectory_analysis as cli

    root, sample, _ = setup_run(tmp_path)
    save(sample, "a", 0, correct=False)
    save(sample, "a", 1)
    monkeypatch.setattr(
        cli,
        "load_trajectory_tokenizer",
        lambda _: (_ for _ in ()).throw(OSError("not cached")),
    )
    monkeypatch.setattr(
        cli,
        "audit_evaluations",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("Saved judgments must be trusted by default")
        ),
    )
    output = tmp_path / "report"
    cli.main([str(root), "--output-dir", str(output)])
    assert (output / "readouts.csv").exists()
    assert (output / "sample_001288.png").exists()
    assert "not cached" in (output / "summary.md").read_text()
    assert (
        json.loads((output / "tensor_status.json").read_text())[0]["status"]
        == "missing"
    )
    assert len(json.loads((output / "review_candidates.json").read_text())) == 1
    # Rerunning never overwrites human annotations.
    annotations = output / "annotations.json"
    payload = json.dumps(
        [
            {
                "source_run_id": "run",
                "sample_id": 1288,
                "condition": "a",
                "step": 0,
                "semantic_correct": True,
                "reasoning_validity": "invalid",
            }
        ]
    )
    annotations.write_text(payload)
    cli.main(
        [
            str(root),
            "--output-dir",
            str(output),
            "--continuations-only",
            "--skip-token-distance",
            "--annotations",
            str(annotations),
        ]
    )
    assert annotations.read_text() == payload
    assert "Human-reviewed readouts: 1" in (output / "summary.md").read_text()
    import csv

    with (output / "readouts.csv").open() as stream:
        reviewed = next(csv.DictReader(stream))
    assert reviewed["semantic_correct"] == "True"
    assert reviewed["evaluator_correct"] == "False"
    assert reviewed["reasoning_validity"] == "invalid"


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
        "tokenizer": "Model",
        "model_revision": "saved-commit",
        "state_identity": {"tokenizer": hashlib.sha256(b"tokenizer-data").hexdigest()},
    }
    assert adapter.load_trajectory_tokenizer(metadata) is tokenizer
    loader.assert_called_once_with(
        "Model", revision="saved-commit", local_files_only=True
    )
    metadata["state_identity"]["tokenizer"] = "different"
    with pytest.raises(ValueError, match="differs"):
        adapter.load_trajectory_tokenizer(metadata)


def test_alignment_pairs_only_shared_complete_items_and_checks_protocol(tmp_path):
    from latentmas_reprop.application.latent_alignment_analysis import compare_alignment

    root, sample, metadata = setup_run(tmp_path)
    save(sample, "a", 0)
    rows, _ = load_readouts(root, metadata)
    other = {**metadata, "latent_space_realign": True}
    paired = compare_alignment(rows, rows, metadata, other)
    assert paired[0]["n_pairs"] == 1
    assert paired[1]["status"] == "pending"
    assert paired[1]["accuracy_diff"] is None
    incompatible = compare_alignment(rows, rows, metadata, {**other, "seed": 999})
    assert incompatible[0]["status"] == "incompatible"
    assert incompatible[0]["n_pairs"] == 0
    assert "seed" in incompatible[0]["reason"]
