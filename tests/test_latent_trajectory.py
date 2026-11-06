"""Trajectory collection preserves latent prefixes across independent readouts."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from latentmas_reprop.application.latent_trajectory import collect, prepare_prompt
from latentmas_reprop.application.latent_trajectory_analysis import trajectory_metrics
from tests.test_intervention_continuation import _build_mock_wrapper


def test_collection_multiple_samples_and_resume(tmp_path, monkeypatch):
    bars = []

    def progress_bar(**kwargs):
        bar = MagicMock()
        bar.__enter__.return_value = bar
        bars.append((kwargs, bar))
        return bar

    monkeypatch.setattr(
        "latentmas_reprop.application.latent_trajectory.tqdm", progress_bar
    )
    model = _build_mock_wrapper()
    from latentmas_reprop.application.intervention_continuation.decoder import (
        decode_tokens_with_hiddens,
    )

    forward = model.model.forward
    latent_calls = []

    def checked_forward(*args, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            latent_calls.append(1)
        cache = kwargs.get("past_key_values")
        if cache is not None:
            assert torch.all(cache[0][0] == 1)
        return forward(*args, **kwargs)

    model.model.forward = checked_forward

    order = []

    def mutating_decode(*args, **kwargs):
        order.append(kwargs["desc"])
        assert kwargs["progress_position"] == 2
        assert len(list(tmp_path.glob("samples/*/trajectories/*/kv.pt"))) >= 2
        assert args[2][0][0].shape[-2] == int(args[3][0]) + 1
        result = decode_tokens_with_hiddens(*args, **kwargs)
        args[2][0][0].fill_(999)
        return result

    monkeypatch.setattr(
        "latentmas_reprop.application.latent_trajectory_readout.decode_tokens_with_hiddens",
        mutating_decode,
    )
    model.tokenizer.unk_token_id = None
    model.tokenizer.all_special_ids = [20, 21]
    model.tokenizer.convert_tokens_to_ids = lambda t: {"<think>": 20, "</think>": 21}[t]
    model.tokenizer.encode = lambda t, **kw: [model.tokenizer.convert_tokens_to_ids(t)]
    monkeypatch.setattr(
        "latentmas_reprop.application.latent_trajectory.prepare_prompt",
        lambda *a: ("prompt", torch.tensor([[1, 2, 3]])),
    )
    args = SimpleNamespace(
        task="gsm8k",
        split="test",
        max_samples=2,
        latent_steps=3,
        latent_step_start=1,
        conditions=list("abcd"),
        seed=42,
        max_new_tokens=1,
        temperature=0.0,
        top_p=0.95,
        no_progress=True,
        latent_space_realign=False,
        initial_input="rq",
        initial_role="direct_solve",
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": str(i), "gold": "4"} for i in range(3)]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("4", True, None)
    completed_steps = []
    assert (
        collect(
            model,
            args,
            dataset,
            evaluator,
            tmp_path,
            sample_context=lambda d: nullcontext(
                lambda k: completed_steps.append((d.name, k, len(order)))
            ),
        )
        == 2
    )
    assert completed_steps == [
        (f"{i:06d}", k, i * 12 + k * 4) for i in range(2) for k in range(1, 4)
    ]
    assert order[:4] == [f"Sample 000000 · {c} · latent 1/3" for c in "ABCD"]
    assert all(
        options["total"] == 3 for options, _ in bars if options["unit"] == "step"
    )
    assert all(
        bar.update.call_count == 3 for options, bar in bars if options["unit"] == "step"
    )
    assert bars[0][0]["unit"] == "sample"
    assert bars[0][0]["total"] == 2
    assert bars[0][0]["position"] == 0
    assert all(
        options["position"] == 1 for options, _ in bars if options["unit"] == "step"
    )
    assert bars[0][1].update.call_count == 2
    assert len(list(tmp_path.glob("samples/*/continuations/*/*.json"))) == 24
    assert len(list(tmp_path.glob("samples/*/trajectories/*.pt"))) == 4
    assert len(list(tmp_path.glob("samples/*/continuations/*/*/hidden.pt"))) == 24
    for sample in ("000000", "000001"):
        base = tmp_path / "samples" / sample
        plain = torch.load(base / "trajectories/without_think.pt", weights_only=True)
        opened = torch.load(base / "trajectories/with_think.pt", weights_only=True)
        assert plain["hidden"][:, 0].tolist() == [3.0, 4.0, 5.0]
        assert opened["hidden"][:, 0].tolist() == [4.0, 5.0, 6.0]
        assert plain["logits"].shape == (3, 32)
        for condition in "abcd":
            row = json.loads(
                (base / f"continuations/{condition}/step_001.json").read_text()
            )
            hidden_path = base / f"continuations/{condition}/step_001/hidden.pt"
            generated_hidden = torch.load(hidden_path, weights_only=True)
            assert generated_hidden.shape == (row["content_length"], 8)
            assert row["rng_provenance"]["policy"] == "sample_v2"
            assert row["identity"]["max_new_tokens"] == 1
            assert row["initial_role"] == "direct_solve"
            assert row["inserted_readout_ids"] == ([21] if condition in "cd" else [])
    calls = evaluator.evaluate.call_count
    collect(model, args, dataset, evaluator, tmp_path)
    assert evaluator.evaluate.call_count == calls
    latent_count = len(latent_calls)
    missing = tmp_path / "samples/000000/continuations/c/step_002.json"
    expected = json.loads(missing.read_text())
    missing.unlink()
    collect(model, args, dataset, evaluator, tmp_path)
    assert json.loads(missing.read_text()) == expected
    assert evaluator.evaluate.call_count == calls + 1
    assert len(latent_calls) == latent_count
    assert bars[-3][0]["unit"] == "sample"
    assert bars[-3][0]["initial"] == 1
    assert bars[-3][1].update.call_count == 1
    assert bars[-2][0]["initial"] == 2
    assert bars[-2][1].update.call_count == 1
    assert bars[-1][0]["initial"] == 3
    assert bars[-1][1].update.call_count == 0
    # Older records with no generated hiddens are replayed, then verified and completed.
    hidden_path = missing.with_suffix("") / "hidden.pt"
    hidden_path.unlink()
    collect(model, args, dataset, evaluator, tmp_path)
    assert hidden_path.exists()
    assert json.loads(missing.read_text()) == expected
    assert len(latent_calls) == latent_count
    for kv_path in tmp_path.glob("samples/*/trajectories/*/kv.pt"):
        snapshot = torch.load(kv_path, weights_only=False)
        assert torch.all(snapshot["past_key_values"][0][0] == 1)
    args.max_samples = 1
    assert collect(model, args, dataset, evaluator, tmp_path) == 1
    args.sample_indices = [2, 0, 2]
    calls = evaluator.evaluate.call_count
    previous = (tmp_path / "samples/000001/sample.json").read_bytes()
    assert collect(model, args, dataset, evaluator, tmp_path) == 2
    assert evaluator.evaluate.call_count == calls + 12
    assert (tmp_path / "samples/000001/sample.json").read_bytes() == previous
    selected = json.loads((tmp_path / "samples/000002/sample.json").read_text())
    from latentmas_reprop.application.intervention_continuation.identity import (
        sample_seed,
    )

    assert selected["index"] == 2
    assert selected["question"] == "2"
    assert selected["seed"] == sample_seed(42, "gsm8k", "test", 2, "2")


def test_template_strips_only_automatic_opening():
    model = _build_mock_wrapper()
    model.prepare_chat_batch = lambda *a, **kw: (
        ["user <think> text\nassistant\n<think>\n"],
        None,
        None,
        None,
    )
    model.tokenizer.apply_chat_template = lambda *a, **kw: "user <think> text\n"
    original = model.tokenizer

    class Tokenizer:
        def __getattr__(self, name):
            return getattr(original, name)

        def __call__(self, text, **kwargs):
            if kwargs.get("return_tensors"):
                return original(text, **kwargs)
            return {"input_ids": [10, 11]}

    model.tokenizer = Tokenizer()
    prompt, _ = prepare_prompt(model, [], [20, 21])
    assert prompt == "user <think> text\nassistant\n"


def test_analysis_constant_distribution_and_direction():
    data = {
        "hidden": torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]),
        "logits": torch.zeros(3, 12),
        "steps": torch.arange(1, 4),
    }
    rows = trajectory_metrics(data)
    assert rows[0]["adjacent_cosine"] is None
    assert rows[1]["js_divergence"] == 0
    assert rows[2]["direction_cosine"] == 1
    assert rows[2]["adjacent_l2"] == 1
