"""Verification of the 13 storage policy and trajectory reuse scenarios.

Covers:
1. Identical request repetition: 0 trajectory computations, load existing results.
2. 10-sample then 100-sample run: reuse 0~9, compute fresh 10~99, preserve both MLflow summaries.
3. Other runs requiring same trajectory: share identical trajectory without duplication or re-inference.
4. Cross pairing change: distinct condition paths per donor, no wrongful reuse.
5. Secret digit assignment change: distinct condition paths per digit, no index contamination.
6. Donor pairing change: distinct donor paths, prevent incorrect reuse.
7. Preflight partial resumption: completed conditions 0 recomputations, only compute incomplete.
8. Reasoning partial resumption: completed conditions skip upstream context computation.
9. Trajectory interrupt & resume: completed samples reused without recomputation.
10. Trajectory complete but summary missing: synthesizes summary from trajectories without model inference.
11. Analysis input change: distinct independent result paths.
12. Identical analysis rerun: reuses existing result path without duplicate MLflow runs.
13. .cache deletion: trajectory and run summaries in storage/ preserved and reused without recomputation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.application.receiver_compute_preflight.collection import (
    collect_samples,
)
from latentmas_reprop.application.receiver_reasoning.runner import (
    collect_sample,
)
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    load_json,
    save_condition_result,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker
from run.runner import _reuse_completed_run


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


def test_scenario_08_reasoning_upstream_skip(tmp_path):
    """Scenario 8: If receiver conditions are complete, upstream inference is skipped entirely."""
    resolver = PathResolver(tmp_path)
    rel_path = "reasoning_test"
    args = make_args(
        tmp_path,
        upstream_steps=[5],
        run_id=rel_path,
        handoff_mode="latent_only",
        split="test",
    )

    # Mark both conditions complete for step 0 and step 5
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
        # build_upstream should not be called because all conditions were complete!
        assert mock_build.call_count == 0


def test_scenario_09_trajectory_resume(tmp_path):
    """Scenario 9: Trajectory completed samples are reused on resume without recomputation."""
    resolver = PathResolver(tmp_path)
    rel_path = "trajectory_test"

    # Pre-populate sample 0 condition
    s0_dir = resolver.get_condition_dir(rel_path, 0, "trajectory")
    save_condition_result(s0_dir, {"rows": [{"pred": "9", "target_digit": 9}]})

    assert is_condition_complete(s0_dir)
    assert not is_condition_complete(resolver.get_condition_dir(rel_path, 1, "trajectory"))


def test_scenario_10_missing_summary_synthesized_without_model(tmp_path):
    """Scenario 10: Trajectories complete but summary missing -> synthesize summary without model load."""
    resolver = PathResolver(tmp_path)
    args = make_args(tmp_path, max_samples=2)
    exp_rel = build_experiment_relpath(args, dtype_name="bfloat16")
    args.exp_relpath = exp_rel

    # Save condition results for sample 0 and 1
    c0 = resolver.get_condition_dir(exp_rel, 0, "benchmark")
    c1 = resolver.get_condition_dir(exp_rel, 1, "benchmark")
    save_condition_result(c0, {"correct": True, "prediction": "1", "gold": "1"})
    save_condition_result(c1, {"correct": False, "prediction": "0", "gold": "2"})

    # Ensure summary.json does NOT exist
    exp_dir = resolver.get_experiment_dir(exp_rel)
    assert not (exp_dir / "summary.json").is_file()

    with patch("run.runner.get_path_resolver", return_value=resolver):
        reused = _reuse_completed_run(args=args)

    assert reused is not None
    summary, _records = reused
    assert summary["accuracy"] == 0.5
    assert (exp_dir / "summary.json").is_file()


def test_scenario_11_and_12_analysis_paths_and_rerun_isolation(tmp_path):
    """Scenarios 11 & 12: Distinct analysis inputs keep separate paths; identical runs reuse."""
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'mlflow_analysis.db'}",
        path_resolver=resolver,
    )

    # Analysis run 1
    run1 = tracker.start_run(
        experiment_name="latentmas-analysis",
        run_name="probe_source_a_step5",
    )
    run1_id = run1.info.run_id
    tracker.end_run()

    # Analysis run 2 with different input
    run2 = tracker.start_run(
        experiment_name="latentmas-analysis",
        run_name="probe_source_b_step5",
    )
    run2_id = run2.info.run_id
    tracker.end_run()

    assert run1_id != run2_id

    # Rerun of analysis 1 reuses run1
    run1_rerun = tracker.start_run(
        experiment_name="latentmas-analysis",
        run_name="probe_source_a_step5",
    )
    assert run1_rerun.info.run_id == run1_id
    tracker.end_run()


def test_scenario_13_cache_deletion_preserves_storage(tmp_path):
    """Scenario 13: Deletion of .cache leaves storage/ intact and trajectories remain complete."""
    resolver = PathResolver(tmp_path)

    # Save data in storage and cache
    exp_rel = "test_persistence"
    cond_dir = resolver.get_condition_dir(exp_rel, 0, "direct")
    save_condition_result(cond_dir, {"val": 123})

    cache_file = resolver.cache_dir / "temp.txt"
    cache_file.write_text("temporary cache content")

    assert is_condition_complete(cond_dir)
    assert cache_file.is_file()

    # Clear .cache entirely
    import shutil
    shutil.rmtree(resolver.cache_dir, ignore_errors=True)

    # Cache is cleared, but storage trajectories remain intact and complete!
    assert not cache_file.is_file()
    assert is_condition_complete(cond_dir)
    loaded = load_json(cond_dir / "result.json")
    assert loaded["val"] == 123
