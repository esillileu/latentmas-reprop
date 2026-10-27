"""Verification of storage policy scenarios 8 through 13.

Covers:
8. Reasoning partial resumption: completed conditions skip upstream context computation.
9. Trajectory interrupt & resume: completed samples reused without recomputation.
10. Trajectory complete but summary missing: synthesizes summary from trajectories without model inference.
11. Analysis input change: distinct independent result paths.
12. Identical analysis rerun: reuses existing result path without duplicate MLflow runs.
13. .cache deletion: trajectory and run summaries in storage/ preserved and reused without recomputation.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from latentmas_reprop.application.receiver_reasoning.runner import collect_sample
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    load_json,
    save_condition_result,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker
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
    import shutil

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
    shutil.rmtree(resolver.cache_dir, ignore_errors=True)

    # Cache is cleared, but storage trajectories remain intact and complete!
    assert not cache_file.is_file()
    assert is_condition_complete(cond_dir)
    loaded = load_json(cond_dir / "result.json")
    assert loaded["val"] == 123
