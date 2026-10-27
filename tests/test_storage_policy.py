"""Unified storage policy, resumption, caching, and integrity verification."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.receiver_compute_preflight.collection import (
    collect_samples,
)
from latentmas_reprop.application.receiver_reasoning.runner import collect_sample
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
)
from latentmas_reprop.infrastructure.storage import (
    save_condition_result,
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
        "tracking_experiment_name": "test_exp",
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


def test_scenario_07_preflight_partial_resumption(tmp_path):
    """Scenario 7: Completed conditions in preflight are skipped."""
    resolver = PathResolver(tmp_path)
    rel_path = "preflight_test"
    args = make_args(tmp_path, upstream_steps=[5], run_id=rel_path)

    nh_dir0 = resolver.get_condition_dir(tmp_path, 0, "no_handoff")
    nh_dir1 = resolver.get_condition_dir(tmp_path, 1, "no_handoff")
    save_condition_result(
        nh_dir0,
        {
            "rows": [
                {
                    "sample_id": "sample_0",
                    "handoff_condition": "no_handoff",
                    "accuracy": 1.0,
                    "prediction": "42",
                }
            ]
        },
    )
    save_condition_result(
        nh_dir1,
        {
            "rows": [
                {
                    "sample_id": "sample_1",
                    "handoff_condition": "no_handoff",
                    "accuracy": 1.0,
                    "prediction": "99",
                }
            ]
        },
    )

    model = MagicMock()
    mock_collect = MagicMock(
        side_effect=[
            (
                (None, None, [], {"handoff_positions": 10}, None),
                [
                    {
                        "accuracy": 1.0,
                        "handoff_condition": "matched",
                        "sample_id": "sample_0",
                        "upstream_steps": 5,
                    }
                ],
            ),
            (
                (None, None, [], {"handoff_positions": 12}, None),
                [
                    {
                        "accuracy": 1.0,
                        "handoff_condition": "matched",
                        "sample_id": "sample_1",
                        "upstream_steps": 5,
                    }
                ],
            ),
            (
                (None, None, [], {"handoff_positions": 10}, None),
                [
                    {
                        "accuracy": 1.0,
                        "handoff_condition": "mismatched",
                        "sample_id": "sample_0",
                        "upstream_steps": 5,
                    }
                ],
            ),
            (
                (None, None, [], {"handoff_positions": 12}, None),
                [
                    {
                        "accuracy": 1.0,
                        "handoff_condition": "mismatched",
                        "sample_id": "sample_1",
                        "upstream_steps": 5,
                    }
                ],
            ),
        ]
    )

    with (
        patch(
            "latentmas_reprop.application.receiver_compute_preflight.collection.get_path_resolver",
            return_value=resolver,
        ),
        patch(
            "latentmas_reprop.application.receiver_compute_preflight.collection.collect_condition",
            mock_collect,
        ),
    ):
        collect_samples(
            model=model,
            args=args,
            items=[{"question": "Q1", "gold": "42"}, {"question": "Q2", "gold": "99"}],
            ids=["sample_0", "sample_1"],
            config={},
            tracker=None,
            evaluator=MagicMock(),
            runtime=MagicMock(),
            records=[],
            manifest={},
            records_path=tmp_path / "records.jsonl",
        )

        called_conditions = [call.args[6] for call in mock_collect.call_args_list]
        assert "no_handoff" not in called_conditions
        assert "matched" in called_conditions


def test_scenario_08_reasoning_upstream_skip(tmp_path):
    """Scenario 8: If receiver conditions are complete, upstream inference is skipped."""
    resolver = PathResolver(tmp_path)
    rel_path = "reasoning_test"
    args = make_args(
        tmp_path,
        upstream_steps=[5],
        run_id=rel_path,
        handoff_mode="latent_only",
        split="test",
    )

    for s in [0, 5]:
        for mode in ["answer_only", "free"]:
            c = resolver.get_condition_dir(tmp_path, 0, f"u{s}_{mode}")
            save_condition_result(
                c,
                {
                    "prediction": "1",
                    "gold": "1",
                    "correct": True,
                    "empty_output": False,
                    "context_id": f"ctx_{s}_{mode}",
                    "upstream_latent_steps": s,
                    "receiver_mode": mode,
                    "receiver_token_limit_reached": False,
                    "parse_failure": False,
                    "execution_success": True,
                    "receiver_latency_sec": 0.01,
                    "upstream_build_latency_sec": 0.01,
                    "receiver_generated_tokens": 1,
                    "receiver_prompt_tokens": 1,
                    "upstream_prompt_tokens": 1,
                    "upstream_latent_steps_executed": 0,
                },
            )

    model = MagicMock()
    records = []

    with (
        patch(
            "latentmas_reprop.application.receiver_reasoning.runner.get_path_resolver",
            return_value=resolver,
        ),
        patch(
            "latentmas_reprop.application.receiver_reasoning.runner.build_upstream",
        ) as mock_build,
    ):
        collect_sample(
            model=model,
            args=args,
            item={"question": "Q", "gold": "1"},
            index=0,
            config={},
            records=records,
            records_path=tmp_path / "records.jsonl",
            evaluator_port=MagicMock(),
            tracker_port=None,
            runtime=MagicMock(),
        )
        assert len(records) == 4
        assert mock_build.call_count == 0
