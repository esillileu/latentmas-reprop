"""Tests for simplified file-based execution storage and contracts."""

import json

import pytest
import torch

from latentmas_reprop.domain.models.tensor_artifact import (
    ConfigurationMismatchError,
)
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    FileStorage,
    ResearchDataReader,
    is_condition_complete,
    save_condition_result,
    save_tensor,
    verify_run_config,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker


def test_1_two_samples_same_condition_no_overwrite(tmp_path):
    """1. 두 샘플의 동일 조건 결과가 서로 덮어쓰지 않는가?"""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)

    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0])

    storage.save_condition(
        run_id="run_test",
        sample_key="sample_A",
        condition="own",
        result_data={"sample": "A", "val": 10},
        tensors={"hidden": t1},
    )
    storage.save_condition(
        run_id="run_test",
        sample_key="sample_B",
        condition="own",
        result_data={"sample": "B", "val": 20},
        tensors={"hidden": t2},
    )

    res_a = storage.load_condition_result("run_test", "sample_A", "own")
    res_b = storage.load_condition_result("run_test", "sample_B", "own")

    assert res_a["val"] == 10
    assert res_b["val"] == 20

    ten_a = storage.load_condition_tensor("run_test", "sample_A", "own", "hidden")
    ten_b = storage.load_condition_tensor("run_test", "sample_B", "own", "hidden")

    assert torch.equal(ten_a, t1)
    assert torch.equal(ten_b, t2)


def test_2_condition_a_completed_condition_b_interrupted_reuse_a_only(tmp_path):
    """2. 조건 A가 완료되고 조건 B에서 중단되면 A만 재사용하는가?"""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)

    # Complete condition A
    storage.save_condition(
        run_id="run_partial",
        sample_key="s1",
        condition="own",
        result_data={"step": "A", "done": True},
        tensors={"hidden": torch.tensor([1.0])},
    )

    # Incomplete condition B (no result.json)
    dir_b = storage.get_condition_dir("run_partial", "s1", "cross")
    dir_b.mkdir(parents=True, exist_ok=True)
    (dir_b / "hidden.pt.tmp").write_text("partial")

    assert storage.is_condition_complete("run_partial", "s1", "own") is True
    assert storage.is_condition_complete("run_partial", "s1", "cross") is False

    # Mock rerun logic: check complete condition is reused without recomputing
    recomputed = []
    for cond in ["own", "cross"]:
        if storage.is_condition_complete("run_partial", "s1", cond):
            rec = storage.load_condition_result("run_partial", "s1", cond)
            assert rec["step"] == "A"
        else:
            recomputed.append(cond)
            storage.save_condition(
                "run_partial", "s1", cond, {"step": "B", "done": True}
            )

    assert recomputed == ["cross"]
    assert storage.is_condition_complete("run_partial", "s1", "cross") is True


def test_3_missing_result_json_not_complete(tmp_path):
    """3. result.json이 없으면 완료 상태로 판단하지 않는가?"""
    resolver = PathResolver(tmp_path)
    cond_dir = resolver.get_condition_dir("run_x", "s1", "own")
    cond_dir.mkdir(parents=True, exist_ok=True)
    save_tensor(cond_dir / "hidden.pt", torch.tensor([1.0]))

    assert is_condition_complete(cond_dir) is False


def test_4_missing_hidden_file_reexecution(tmp_path):
    """4. Hidden 파일이 누락된 경우 해당 조건을 재실행하는가?"""
    resolver = PathResolver(tmp_path)
    cond_dir = resolver.get_condition_dir("run_x", "s1", "own")
    cond_dir.mkdir(parents=True, exist_ok=True)

    # result.json declares hidden.pt, but hidden.pt does not exist
    result_path = cond_dir / "result.json"
    result_path.write_text(
        json.dumps({"files": ["hidden.pt"], "status": "ok"}), encoding="utf-8"
    )

    assert is_condition_complete(cond_dir) is False


def test_5_different_config_resume_rejected(tmp_path):
    """5. 다른 설정으로 기존 Run을 재개하려 하면 거부하는가?"""
    resolver = PathResolver(tmp_path)
    run_dir = resolver.get_experiment_dir("run_cfg_test")

    initial_cfg = {
        "model_name": "Qwen/Qwen3-4B",
        "task": "gsm8k",
        "method": "latent_mas",
        "latent_steps": 20,
        "seed": 42,
    }
    verify_run_config(run_dir, initial_cfg)

    # Same config should pass cleanly
    verify_run_config(run_dir, initial_cfg)

    # Different config should be rejected
    mismatched_cfg = dict(initial_cfg)
    mismatched_cfg["latent_steps"] = 40

    with pytest.raises(ConfigurationMismatchError, match="Configuration mismatch"):
        verify_run_config(run_dir, mismatched_cfg)


def test_6_atomic_save_failure_leaves_no_completion_file(tmp_path, monkeypatch):
    """6. 원자적 저장 중 실패하면 완료 파일이 생성되지 않는가?"""
    resolver = PathResolver(tmp_path)
    cond_dir = resolver.get_condition_dir("run_err", "s1", "own")

    def failing_save(*args, **kwargs):
        raise OSError("Simulated disk write failure")

    monkeypatch.setattr(torch, "save", failing_save)

    with pytest.raises(OSError, match="Simulated disk write failure"):
        save_condition_result(
            cond_dir,
            result_data={"valid": False},
            tensors={"hidden": torch.tensor([1.0])},
        )

    assert not (cond_dir / "result.json").exists()
    assert is_condition_complete(cond_dir) is False


def test_7_clean_cache_preserves_research_storage(tmp_path):
    """7. clean-cache 이후에도 모든 연구 원본이 보존되는가?"""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)
    manager = ExecutionCacheManager(resolver)

    # Save research original in storage/experiments
    res_path = storage.save_condition(
        run_id="run_research",
        sample_key="s1",
        condition="own",
        result_data={"score": 1.0},
        tensors={"hidden": torch.tensor([42.0])},
    )
    assert res_path.is_file()

    # Create volatile cache files in .cache/runtime
    runtime_file = manager.get_layer_path("runtime") / "scratch.json"
    runtime_file.write_text("scratch", encoding="utf-8")

    manager.clean_volatile_cache()

    assert not runtime_file.exists()
    assert res_path.is_file()
    assert is_condition_complete(res_path.parent) is True
    assert torch.equal(
        storage.load_condition_tensor("run_research", "s1", "own", "hidden"),
        torch.tensor([42.0]),
    )


def test_8_mlflow_run_finds_local_data_dir(tmp_path):
    """8. MLflow Run에서 해당 데이터 디렉터리를 찾아갈 수 있는가?"""
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(path_resolver=resolver)

    run = tracker.start_run("test_exp", run_name="exp_1")
    run_id = run.info.run_id

    reader = ResearchDataReader(mlflow_client=tracker.client, path_resolver=resolver)
    found_dir = reader.get_run_data_dir(run_id)

    assert found_dir == resolver.get_experiment_dir(run_id)
    assert found_dir.is_dir()
    tracker.end_run()


def test_9_resume_same_run_no_duplicate_results(tmp_path):
    """9. 동일 Run을 재개해도 정상 결과가 중복 생성되지 않는가?"""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)

    t = torch.tensor([5.0])
    storage.save_condition(
        run_id="run_idempotent",
        sample_key="s1",
        condition="own",
        result_data={"count": 1},
        tensors={"hidden": t},
    )

    cond_dir = storage.get_condition_dir("run_idempotent", "s1", "own")
    mtime_before = (cond_dir / "result.json").stat().st_mtime_ns

    # Calling save_condition again without overwrite should skip re-writing
    save_condition_result(
        cond_dir,
        result_data={"count": 2},
        tensors={"hidden": t},
        overwrite=False,
    )
    mtime_after = (cond_dir / "result.json").stat().st_mtime_ns

    assert mtime_before == mtime_after
    res = storage.load_condition_result("run_idempotent", "s1", "own")
    assert res["count"] == 1


def test_10_migrated_data_queried_by_new_reader(tmp_path):
    """10. 마이그레이션 도구로 생성한 데이터가 새 Reader에서 조회되는가?"""
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(path_resolver=resolver)

    run_id = "migrated_run_10"
    exp_dir = resolver.get_experiment_dir(run_id)

    # Simulate migrated benchmark summary and tensor
    summary_file = exp_dir / "summary.json"
    summary_file.write_text(
        json.dumps({"status": "complete", "accuracy": 0.95}), encoding="utf-8"
    )

    tensor_file = exp_dir / "sender_states.pt"
    save_tensor(tensor_file, torch.tensor([10.0, 20.0]))

    loaded_summary = reader.load_artifact(run_id, "summary.json")
    assert loaded_summary["status"] == "complete"
    assert loaded_summary["accuracy"] == 0.95

    loaded_tensor = reader.load_tensor(run_id, "sender_states")
    assert torch.equal(loaded_tensor, torch.tensor([10.0, 20.0]))
