"""Verification of storage policy scenarios 1 through 7.

Covers:
1. Identical request repetition: 0 trajectory computations, load existing results.
2. 10-sample then 100-sample run: reuse 0~9, compute fresh 10~99, preserve both MLflow summaries.
3. Other runs requiring same trajectory: share identical trajectory without duplication or re-inference.
4. Cross pairing change: distinct condition paths per donor, no wrongful reuse.
5. Secret digit assignment change: distinct condition paths per digit, no index contamination.
6. Donor pairing change: distinct donor paths, prevent incorrect reuse.
7. Preflight partial resumption: completed conditions 0 recomputations, only compute incomplete.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.receiver_compute_preflight.collection import (
    collect_samples,
)
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    save_condition_result,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


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


def test_scenario_01_identical_request_repetition(tmp_path):
    """Scenario 1: Identical request repetition reuses all trajectories with 0 fresh computations."""
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)
    samples = [
        {"question": "Q1", "solution": "A1", "gold": "1"},
        {"question": "Q2", "solution": "A2", "gold": "2"},
    ]
    dataset = DummyDataset(samples)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = True

    model = MagicMock()
    model.model_name = "Qwen/Qwen3-0.6B"
    model.dtype_name = "bfloat16"
    model.device = "cpu"
    model.prepare_chat_batch.return_value = (["prompt"], torch.ones(1, 1), torch.ones(1, 1), [["tok"]])
    model.generate_text_batch.return_value = (
        ["1"],
        [{"text": "1", "raw_output": "1", "error": None}],
        [1],
    )

    uc = BenchmarkUseCase(
        dataset_port=dataset,
        cache_port=cache,
        evaluator_port=evaluator,
    )
    args = make_args(tmp_path, max_samples=2)

    # First run: 2 fresh computations
    metrics1, _records1 = uc.execute(model=model, args=args)
    assert metrics1.requested_trajectories == 2
    assert metrics1.fresh_trajectories == 2
    assert metrics1.reused_trajectories == 0
    assert model.generate_text_batch.call_count == 2

    # Second run: identical request repetition -> 0 fresh computations, 2 reused
    metrics2, _records2 = uc.execute(model=model, args=args)
    assert metrics2.requested_trajectories == 2
    assert metrics2.fresh_trajectories == 0
    assert metrics2.reused_trajectories == 2
    assert model.generate_text_batch.call_count == 2  # No new calls!


def test_scenario_02_and_03_scope_expansion_and_sharing(tmp_path):
    """Scenarios 2 & 3: Run A (2 samples) followed by Run B (4 samples).

    Reuses 0~1 without recomputation, computes 2~3, and preserves both MLflow summaries.
    """
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)
    samples = [
        {"question": f"Q{i}", "solution": f"A{i}", "gold": str(i)}
        for i in range(4)
    ]
    dataset = DummyDataset(samples)
    evaluator = MagicMock()
    evaluator.evaluate.return_value = True

    model = MagicMock()
    model.model_name = "Qwen/Qwen3-0.6B"
    model.dtype_name = "bfloat16"
    model.device = "cpu"
    model.prepare_chat_batch.return_value = (["prompt"], torch.ones(1, 1), torch.ones(1, 1), [["tok"]])
    model.generate_text_batch.return_value = (
        ["ans"],
        [{"text": "ans", "raw_output": "ans", "error": None}],
        [1],
    )

    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        path_resolver=resolver,
    )

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
    assert metrics_b.reused_trajectories == 2  # 0~1 reused!
    assert metrics_b.fresh_trajectories == 2  # 2~3 fresh!
    assert model.generate_text_batch.call_count == 4

    # Both run summaries preserved under distinct run names in MLflow
    exp_rel = build_experiment_relpath(args_a, dtype_name="bfloat16")
    runs = tracker.client.search_runs(
        experiment_ids=[tracker.client.get_experiment_by_name("test_exp").experiment_id]
    )
    run_names = {r.data.tags.get("mlflow.runName") for r in runs}
    assert f"{exp_rel}_n2" in run_names
    assert f"{exp_rel}_n4" in run_names


def test_scenario_04_cross_pairing_isolation(tmp_path):
    """Scenario 4: Cross pairing changes (e.g. shift_1 with different sample counts) do not collide."""
    resolver = PathResolver(tmp_path)
    rel_path = "test_exp_cross"
    cond_dir_donor1 = resolver.get_condition_dir(rel_path, 0, "cross_src_1")
    cond_dir_donor2 = resolver.get_condition_dir(rel_path, 0, "cross_src_2")

    save_condition_result(cond_dir_donor1, {"pred": "ans1", "gold": "gold1"})
    assert is_condition_complete(cond_dir_donor1)
    assert not is_condition_complete(cond_dir_donor2)

    # Cross donor 2 requires fresh computation and does not wrongfully reuse donor 1
    assert cond_dir_donor1 != cond_dir_donor2


def test_scenario_05_secret_digit_assignment_isolation(tmp_path):
    """Scenario 5: Different secret digit assignments for same index do not contaminate each other."""
    resolver = PathResolver(tmp_path)
    rel_path = "secret_digit_exp"

    # Sample 0 with digit 3
    dir_d3 = resolver.get_condition_dir(rel_path, 0, "d3_src3")
    save_condition_result(dir_d3, {"digit": 3, "acc": 1.0})

    # Sample 0 with digit 7
    dir_d7 = resolver.get_condition_dir(rel_path, 0, "d7_src7")
    assert not is_condition_complete(dir_d7)
    assert dir_d3 != dir_d7


def test_scenario_06_preflight_donor_pairing_isolation(tmp_path):
    """Scenario 6: Donor pairing change in compute preflight isolates condition directories."""
    resolver = PathResolver(tmp_path)
    rel_path = "preflight_exp"

    # Mismatched donor index 2 vs 3
    dir_donor2 = resolver.get_condition_dir(rel_path, 0, "u5_mismatched_src2")
    dir_donor3 = resolver.get_condition_dir(rel_path, 0, "u5_mismatched_src3")

    save_condition_result(dir_donor2, {"accuracy": 1.0})
    assert is_condition_complete(dir_donor2)
    assert not is_condition_complete(dir_donor3)
    assert dir_donor2 != dir_donor3


def test_scenario_07_preflight_partial_resumption(tmp_path):
    """Scenario 7: Completed conditions in preflight are skipped (0 recomputations)."""
    resolver = PathResolver(tmp_path)
    rel_path = "preflight_test"
    args = make_args(tmp_path, upstream_steps=[5], run_id=rel_path)

    # Mark no_handoff as already complete for sample 0 and 1
    nh_dir0 = resolver.get_condition_dir(tmp_path, 0, "no_handoff")
    nh_dir1 = resolver.get_condition_dir(tmp_path, 1, "no_handoff")
    save_condition_result(
        nh_dir0,
        {"rows": [{"sample_id": "sample_0", "handoff_condition": "no_handoff", "accuracy": 1.0, "prediction": "42"}]},
    )
    save_condition_result(
        nh_dir1,
        {"rows": [{"sample_id": "sample_1", "handoff_condition": "no_handoff", "accuracy": 1.0, "prediction": "99"}]},
    )

    model = MagicMock()
    mock_collect = MagicMock(
        side_effect=[
            ((None, None, [], {"handoff_positions": 10}, None), [{"accuracy": 1.0, "handoff_condition": "matched", "sample_id": "sample_0", "upstream_steps": 5}]),
            ((None, None, [], {"handoff_positions": 12}, None), [{"accuracy": 1.0, "handoff_condition": "matched", "sample_id": "sample_1", "upstream_steps": 5}]),
            ((None, None, [], {"handoff_positions": 10}, None), [{"accuracy": 1.0, "handoff_condition": "mismatched", "sample_id": "sample_0", "upstream_steps": 5}]),
            ((None, None, [], {"handoff_positions": 12}, None), [{"accuracy": 1.0, "handoff_condition": "mismatched", "sample_id": "sample_1", "upstream_steps": 5}]),
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

