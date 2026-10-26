import torch

from latentmas_reprop.domain.ports.cache_port import CacheLayer
from latentmas_reprop.infrastructure.cache import get_cache_manager


def test_cache_manager_layers():
    manager = get_cache_manager()
    realign_dir = manager.get_layer_path(CacheLayer.MODELS_REALIGN)
    runs_dir = manager.get_layer_path(CacheLayer.EVALUATION_RUNS)

    assert realign_dir.is_dir()
    assert runs_dir.is_dir()
    assert ".cache/models/realign" in str(realign_dir)
    assert ".cache/evaluation/runs" in str(runs_dir)


def test_cache_manager_json_io():
    manager = get_cache_manager()
    test_data = {"task": "gsm8k", "acc": 1.0}
    filename = "test_run.json"

    saved_path = manager.save_json(CacheLayer.RUNTIME, filename, test_data)
    assert saved_path.is_file()
    assert manager.exists(CacheLayer.RUNTIME, filename)

    loaded = manager.load_json(CacheLayer.RUNTIME, filename)
    assert loaded == test_data

    # Cleanup
    saved_path.unlink(missing_ok=True)


def test_cache_manager_torch_io():
    manager = get_cache_manager()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    filename = "test_tensor.pt"

    saved_path = manager.save_torch(CacheLayer.RUNTIME, filename, tensor)
    assert saved_path.is_file()

    loaded = manager.load_torch(CacheLayer.RUNTIME, filename)
    assert torch.equal(loaded, tensor)

    # Cleanup
    saved_path.unlink(missing_ok=True)


def test_clean_volatile_cache(tmp_path):
    from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
    from latentmas_reprop.infrastructure.paths.resolver import PathResolver

    resolver = PathResolver(tmp_path)
    manager = ExecutionCacheManager(resolver)

    # Persistent files must be preserved
    persistent_db = resolver.mlflow_db_path
    persistent_db.parent.mkdir(parents=True, exist_ok=True)
    persistent_db.write_text("dummy db")

    tensor_file = resolver.experiments_dir / "test_run" / "kv.pt"
    tensor_file.parent.mkdir(parents=True, exist_ok=True)
    tensor_file.write_text("dummy tensor")

    exp_file = resolver.experiments_dir / "test_run" / "config.json"
    exp_file.parent.mkdir(parents=True, exist_ok=True)
    exp_file.write_text('{"run_id": "test_run"}')

    # Volatile cache files must be cleaned, but active checkpoints must be preserved
    runtime_file = manager.get_layer_path(CacheLayer.RUNTIME) / "scratch.json"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_text("scratch")

    runtime_checkpoint = (
        manager.get_layer_path(CacheLayer.RUNTIME)
        / "intervention_checkpoint_test.jsonl"
    )
    runtime_checkpoint.write_text('{"sample_key": "k0"}\n')

    # Incomplete benchmark run (resumption jsonl) must be preserved by default
    incomplete_run_dir = (
        manager.get_layer_path(CacheLayer.EVALUATION_RUNS) / "incomplete_run_1"
    )
    incomplete_run_dir.mkdir(parents=True, exist_ok=True)
    resumption_file = incomplete_run_dir / "samples.jsonl"
    resumption_file.write_text('{"sample_key": "k1", "answer": 42}\n')

    cleaned = manager.clean_volatile_cache()
    assert not runtime_file.exists()
    assert runtime_checkpoint.is_file(), (
        "Active runtime checkpoint must be preserved by routine clean-cache"
    )
    assert persistent_db.is_file()
    assert tensor_file.is_file()
    assert exp_file.is_file()
    assert resumption_file.is_file(), (
        "Incomplete benchmark resumption log must not be deleted by routine clean-cache"
    )
    assert isinstance(cleaned, list)

    # Purging explicitly allows removing incomplete runs and checkpoints
    manager.clean_volatile_cache(
        include_resumable_runs=True, purge_incomplete_runs=True
    )
    assert not resumption_file.exists()
    assert not runtime_checkpoint.exists()
