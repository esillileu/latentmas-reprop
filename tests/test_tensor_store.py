"""Tests for file-based execution storage contracts."""

import pytest
import torch

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    FileStorage,
    ResearchDataReader,
    is_condition_complete,
    save_condition_result,
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


def test_8_mlflow_run_finds_local_data_dir(tmp_path):
    """8. MLflow Run에서 해당 데이터 디렉터리를 찾아갈 수 있는가?"""
    resolver = PathResolver(tmp_path)
    tracker = MLflowTracker(path_resolver=resolver)

    run = tracker.start_run("test_exp", run_name="exp_1")
    run_id = run.info.run_id

    reader = ResearchDataReader(mlflow_client=tracker.client, path_resolver=resolver)
    found_dir = reader.get_run_data_dir(run_id)

    assert found_dir == resolver.get_experiment_dir("exp_1")
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
