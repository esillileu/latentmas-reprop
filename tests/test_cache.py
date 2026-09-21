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
