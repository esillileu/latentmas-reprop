"""Reproduction and regression tests for storage policy defects:

1. Sender Context overwrite in receiver trajectory per-sample collection.
2. Run Summary scope mismatch when executing 10 -> 100 -> 10 samples.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from latentmas_reprop.application.receiver_acquisition import trajectory
from latentmas_reprop.application.receiver_acquisition.sampling import (
    generate_secret_digit_samples,
)
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import save_condition_result
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker
from run.runner import _reuse_completed_run


def test_receiver_trajectory_sender_contexts_not_overwritten(tmp_path, monkeypatch):
    """Defect 1: Per-sample trajectory collection must not overwrite sender_contexts.jsonl."""
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
        return SimpleNamespace(logits=torch.arange(32).float()[None], hidden_states=None)

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

    # Sample 0 collection
    s0_file = tmp_path / "sample_0.jsonl"
    trajectory.collect_trajectory(model, [samples[0]], receiver, s0_file, append=True)
    s0_file.unlink(missing_ok=True)

    lines_after_0 = (tmp_path / "sender_contexts.jsonl").read_text().splitlines()
    assert len(lines_after_0) == 1

    # Sample 1 collection (must append, not overwrite sample 0)
    s1_file = tmp_path / "sample_1.jsonl"
    trajectory.collect_trajectory(model, [samples[1]], receiver, s1_file, append=True)
    s1_file.unlink(missing_ok=True)

    lines_after_1 = (tmp_path / "sender_contexts.jsonl").read_text().splitlines()
    assert len(lines_after_1) == 2

    ctx0 = json.loads(lines_after_1[0])
    ctx1 = json.loads(lines_after_1[1])
    assert ctx0["sample_id"] == samples[0].sample_id
    assert ctx1["sample_id"] == samples[1].sample_id


def test_run_summary_scope_aggregation_10_100_10(tmp_path):
    """Defect 2: 10 -> 100 -> 10 run must return summary aggregated for 10 samples, not 100."""
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
        dtype="bfloat16",
        device="cpu",
        overwrite=False,
        tracking_experiment_name="test_exp",
        generate_bs=1,
        use_vllm=False,
    )
    exp_rel = build_experiment_relpath(args_10, dtype_name="bfloat16")
    args_10.exp_relpath = exp_rel

    # Pre-populate 100 completed conditions on disk
    for i in range(100):
        cond_dir = resolver.get_condition_dir(exp_rel, i, "benchmark")
        is_correct = (i % 2 == 0)  # 50 correct out of 100; for first 10: 5 correct
        save_condition_result(cond_dir, {"correct": is_correct, "sample_index": i})

    exp_dir = resolver.get_experiment_dir(exp_rel)

    # Simulate existing summary.json on disk written by 100-sample run
    summary_100 = {
        "status": "complete",
        "sample_count": 100,
        "requested_trajectories": 100,
        "reused_trajectories": 10,
        "fresh_trajectories": 90,
        "failed_trajectories": 0,
        "metrics": {
            "accuracy": 0.5,
            "correct": 50,
            "total": 100,
            "max_samples": 100,
        },
    }
    (exp_dir / "summary.json").write_text(json.dumps(summary_100, indent=2) + "\n")

    # Run for 10 samples
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        path_resolver=resolver,
    )
    with (
        patch("run.runner.get_path_resolver", return_value=resolver),
        patch("run.runner.MLflowTracker", return_value=tracker),
    ):
        reused = _reuse_completed_run(args=args_10)

    assert reused is not None
    metrics, records = reused

    # Crucial assertions: Must reflect 10 requested samples, NOT the 100 on disk!
    assert len(records) == 10
    assert metrics["total"] == 10
    assert metrics["max_samples"] == 10
    assert metrics["sample_count"] == 10
    assert metrics["correct"] == 5
    assert metrics["accuracy"] == 0.5

    # MLflow run must reference canonical exp_rel
    exp = tracker.client.get_experiment_by_name("test_exp")
    runs = tracker.client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags.get("mlflow.runName") == exp_rel

    # On-disk summary.json must remain 100 samples intact
    on_disk_summary = json.loads((exp_dir / "summary.json").read_text())
    assert on_disk_summary["sample_count"] == 100
