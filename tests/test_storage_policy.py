"""Unified storage policy, resumption, caching, and integrity verification."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
)


class DummyDataset:
    def __init__(self, samples):
        self.samples = samples

    def load(self, task, split):
        return iter(self.samples)


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
        "generate_bs": 1,
        "use_vllm": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_scenario_02_and_03_scope_expansion_and_sharing(tmp_path):
    """Scenarios 2 & 3: Run A (2 samples) followed by Run B (4 samples)."""
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)
    samples = [
        {"question": f"Q{i}", "solution": f"A{i}", "gold": str(i)} for i in range(4)
    ]
    dataset = DummyDataset(samples)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = True

    model = MagicMock()
    model.model_name = "Qwen/Qwen3-0.6B"
    model.dtype_name = "bfloat16"
    model.device = "cpu"
    model.prepare_chat_batch.return_value = (
        ["prompt"],
        torch.ones(1, 1),
        torch.ones(1, 1),
        [["tok"]],
    )
    model.generate_text_batch.return_value = (
        ["ans"],
        [{"text": "ans", "raw_output": "ans", "error": None}],
        [1],
    )

    tracker = MagicMock()
    tracker.client = MagicMock()
    exp_rel = build_experiment_relpath(
        make_args(tmp_path, max_samples=2), dtype_name="bfloat16"
    )
    run_mock = MagicMock()
    run_mock.data.tags = {"rel_dir": exp_rel, "mlflow.runName": exp_rel}
    tracker.client.search_runs.return_value = [run_mock]

    uc = BenchmarkUseCase(
        dataset_port=dataset,
        cache_port=cache,
        evaluator_port=evaluator,
        tracker_port=tracker,
    )

    # Run A: 2 samples
    args_a = make_args(tmp_path, max_samples=2)
    metrics_a, _ = uc.execute(model=model, args=args_a)
    assert metrics_a.fresh_trajectories == 2
    assert metrics_a.reused_trajectories == 0
    assert model.generate_text_batch.call_count == 2

    # Run B: 4 samples
    args_b = make_args(tmp_path, max_samples=4)
    metrics_b, _ = uc.execute(model=model, args=args_b)
    assert metrics_b.requested_trajectories == 4
    assert metrics_b.reused_trajectories == 2
    assert metrics_b.fresh_trajectories == 2
    assert model.generate_text_batch.call_count == 4
