"""Unified storage policy, resumption, caching, and integrity verification - Part 2."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from latentmas_reprop.application.receiver_acquisition import trajectory
from latentmas_reprop.application.receiver_acquisition.runner import (
    run_acquisition_loop,
)
from latentmas_reprop.application.receiver_acquisition.sampling import (
    SecretDigitSample,
    generate_secret_digit_samples,
)
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
    get_path_resolver,
)
from latentmas_reprop.infrastructure.storage import (
    ResearchDataReader,
    save_condition_result,
)
from run.runner import _reuse_completed_run


def make_args(tmp_path, max_samples=2, **kwargs):
    defaults = {
        "model_name": "Qwen/Qwen3-0.6B",
        "task": "gsm8k",
        "method": "baseline",
        "split": "test",
        "prompt": "sequential",
        "seed": 42,
        "max_samples": max_samples,
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "latent_steps": 5,
        "backend": "mock",
        "dtype": "bfloat16",
        "run_id": None,
        "overwrite": False,
        "save_raw_cache": False,
        "save_hidden_states": False,
        "tracking_experiment_name": "test_exp",
        "generate_bs": 1,
        "use_vllm": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_scenario_10_missing_summary_synthesized_without_model(tmp_path):
    """Scenario 10: Trajectories complete but summary missing -> synthesize summary without model."""
    resolver = PathResolver(tmp_path)
    args = make_args(tmp_path, max_samples=2)
    exp_rel = build_experiment_relpath(args, dtype_name="bfloat16")
    args.exp_relpath = exp_rel

    c0 = resolver.get_condition_dir(exp_rel, 0, "benchmark")
    c1 = resolver.get_condition_dir(exp_rel, 1, "benchmark")
    save_condition_result(c0, {"correct": True, "prediction": "1", "gold": "1"})
    save_condition_result(c1, {"correct": False, "prediction": "0", "gold": "2"})

    exp_dir = resolver.get_experiment_dir(exp_rel)
    assert not (exp_dir / "summary.json").is_file()

    with patch("run.runner.get_path_resolver", return_value=resolver):
        reused = _reuse_completed_run(args=args)

    assert reused is not None
    summary, _records = reused
    assert summary["accuracy"] == 0.5
    assert (exp_dir / "summary.json").is_file()


def test_receiver_trajectory_sender_contexts_not_overwritten(tmp_path, monkeypatch):
    """Trajectory collection must not overwrite sender_contexts.jsonl."""
    samples = generate_secret_digit_samples(2, 42)
    full = ((torch.zeros(1, 1, 23, 1), torch.ones(1, 1, 23, 1)),)

    def mock_build(model, args, target):
        return SimpleNamespace(
            full=full,
            prompt_len=3,
            full_len=23,
            build_latency_sec=0.01,
            prompt_input={
                "messages": [{"role": "user", "content": f"digit {target.digit}"}],
                "prompt": f"sender rendered digit {target.digit}",
                "input_ids": [1, 2, 3],
                "tokens": ["1", "2", "3"],
                "attention_mask": [1, 1, 1],
            },
        )

    def mock_forward(*args, **kwargs):
        return SimpleNamespace(
            logits=torch.arange(32).float()[None], hidden_states=None
        )

    model = SimpleNamespace(
        model_name="Qwen/Qwen3-0.6B",
        device="cpu",
        forward_next_token_batch=mock_forward,
        tokenizer=SimpleNamespace(convert_ids_to_tokens=str, decode=str),
    )
    receiver = (
        torch.tensor([[1, 2]]),
        torch.ones(1, 2),
        {
            "messages": [{"role": "user", "content": "receiver question"}],
            "prompt": "receiver rendered The number is ",
            "chat_template_kwargs": {"enable_thinking": False},
            "answer_prefix": "The number is ",
            "candidates": {str(d): {"contextual_token_id": d} for d in range(10)},
        },
    )
    monkeypatch.setattr(trajectory, "build_sender_cache", mock_build)

    s0_file = tmp_path / "sample_0.jsonl"
    trajectory.collect_trajectory(model, [samples[0]], receiver, s0_file, append=True)
    s0_file.unlink(missing_ok=True)

    lines_after_0 = (tmp_path / "sender_contexts.jsonl").read_text().splitlines()
    assert len(lines_after_0) == 1

    s1_file = tmp_path / "sample_1.jsonl"
    trajectory.collect_trajectory(model, [samples[1]], receiver, s1_file, append=True)
    s1_file.unlink(missing_ok=True)

    lines_after_1 = (tmp_path / "sender_contexts.jsonl").read_text().splitlines()
    assert len(lines_after_1) == 2


def test_run_summary_scope_aggregation_10_100_10(tmp_path):
    """10 -> 100 -> 10 run must return summary aggregated for 10 samples, not 100."""
    resolver = PathResolver(tmp_path)
    args_10 = SimpleNamespace(
        model_name="Qwen/Qwen3-0.6B",
        task="gsm8k",
        method="baseline",
        split="test",
        prompt="sequential",
        seed=42,
        max_samples=10,
        max_new_tokens=16,
        temperature=0.0,
        top_p=1.0,
        latent_steps=5,
        backend="mock",
        dtype="bfloat16",
        run_id=None,
        overwrite=False,
        save_raw_cache=False,
        save_hidden_states=False,
        tracking_experiment_name="test_exp",
        generate_bs=1,
        use_vllm=False,
    )
    exp_rel = build_experiment_relpath(args_10, dtype_name="bfloat16")
    args_10.exp_relpath = exp_rel

    for i in range(100):
        c = resolver.get_condition_dir(exp_rel, i, "benchmark")
        save_condition_result(c, {"correct": i < 10, "prediction": "1", "gold": "1"})

    exp_dir = resolver.get_experiment_dir(exp_rel)
    (exp_dir / "summary.json").unlink(missing_ok=True)

    with patch("run.runner.get_path_resolver", return_value=resolver):
        reused = _reuse_completed_run(args=args_10)

    assert reused is not None
    summary, _records = reused
    assert summary["sample_count"] == 10
    assert summary["accuracy"] == 1.0


def test_acquisition_lazy_cache_build():
    """Acquisition loop builds sender cache only when needed."""
    resolver = get_path_resolver()
    samples = [
        SecretDigitSample(sample_id="s0", sample_index=0, sample_key="s0", digit=1),
        SecretDigitSample(sample_id="s1", sample_index=1, sample_key="s1", digit=2),
    ]
    c_dir = resolver.get_condition_dir("run_acq", 0, "full_own")
    save_condition_result(
        c_dir,
        {
            "sample_id": "s0",
            "condition": "own",
            "context_mode": "full",
            "target_digit": 1,
        },
    )

    args = SimpleNamespace(
        acquisition_conditions=["own"],
        context_modes=["full"],
        latent_steps=5,
        save_hidden_states=False,
    )
    model = MagicMock()
    model.device = torch.device("cpu")

    rec_mock = MagicMock()
    rec_mock.hidden_state_artifact_key = None
    rec_mock.to_dict.return_value = {
        "sample_id": "s1",
        "condition": "own",
        "context_mode": "full",
    }
    rec_mock.sample_index = 1
    rec_mock.predicted_digit = 2
    rec_mock.source_probability = 0.9
    rec_mock.content_follow_correct = True
    rec_mock.error = None

    with (
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.build_sender_cache"
        ) as mock_build,
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.forward_receiver_observation",
            return_value=(rec_mock, None),
        ),
    ):
        bundle_mock = MagicMock()
        bundle_mock.build_latency_sec = 0.05
        bundle_mock.full = None
        mock_build.return_value = bundle_mock

        records, _, _ = run_acquisition_loop(
            model=model,
            args=args,
            samples=samples,
            cross_indices=[0, 0],
            receiver_ids=torch.zeros((1, 2), dtype=torch.long),
            receiver_mask=torch.ones((1, 2), dtype=torch.long),
            mapping={},
            tracker_port=None,
            run_id="run_acq",
        )
        assert mock_build.call_count == 1
        assert len(records) == 2


def test_reader_path_traversal_blocked(tmp_path):
    """Path traversal characters are strictly rejected with ValueError."""
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(path_resolver=resolver)

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_artifact("run_1", "../secret.json")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="../bad", condition="own")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="s1", condition="../bad")
