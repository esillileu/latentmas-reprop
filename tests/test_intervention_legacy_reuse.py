"""Resume current inference from validated migrated shared checkpoints."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation.checkpoint import (
    _sample_identity,
)
from latentmas_reprop.application.intervention_continuation.runner import (
    InterventionContinuationRunner,
)
from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage
from tools.migrations.intervention_continuation import migrate


@pytest.mark.parametrize(
    "damage, forwards, dynamic, partial",
    [
        (None, 5, False, False),
        ("hidden", 10, False, False),
        ("prefill", 9, False, False),
        (None, 5, True, False),
        (None, 4, False, True),
    ],
)
def test_migrated_state_is_checked_and_resumed(
    tmp_path, monkeypatch, damage, forwards, dynamic, partial
):
    model = _build_mock_wrapper()
    model.model.forward = MagicMock(wraps=model.model.forward)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=5,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_samples=1,
        no_progress=True,
        intervention_conditions=["ple"],
    )
    item = {"question": "Q", "gold": "1"}
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    runner = InterventionContinuationRunner(model, args, evaluator)
    kv, hidden, meta = runner.compute_common_latent(item)
    meta.update(
        _sample_identity(model, args, item, 0),
        status="complete",
        files=["kv.pt", "hidden.pt"],
    )
    if damage == "hidden":
        hidden[-1, 0] += 1
    if damage == "prefill":
        kv[0][0][..., 0, 0] += 1
    if dynamic:
        from transformers.cache_utils import DynamicCache

        kv = DynamicCache(ddp_cache_data=kv)
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)
    root = storage.base_dir
    source = root / "gsm8k/mock/latent_mas/sequential/ls5_intervention_continuation"
    shared = source / "samples/000000/shared"
    shared.mkdir(parents=True)
    (source / "config.json").write_text(
        json.dumps(
            {"model_name": args.model_name, "task": args.task, "latent_steps": 5}
        )
    )
    (shared / "result.json").write_text(json.dumps(meta))
    torch.save(kv, shared / "kv.pt")
    torch.save(hidden, shared / "hidden.pt")
    monkeypatch.setattr(migrate, "find_runs", lambda *a: {})
    report = migrate.migrate(root, "mock://tracking", True)
    assert report["samples"][0]["status"] == "migrated"
    migrated = root / report["samples"][0]["state_path"]
    old_bytes = (migrated / "samples/000000/shared/ls5/hidden.pt").read_bytes()
    dataset = MagicMock()
    dataset.load.return_value = [item]
    uc = InterventionContinuationUseCase(
        dataset_port=dataset, evaluator_port=evaluator, storage=storage
    )
    if partial:
        marker = migrated / "state.json"
        hidden_marker = migrated / "state.json.hidden"
        marker.rename(hidden_marker)
        args.latent_steps = 2
        uc.execute(model, args)
        hidden_marker.rename(marker)
    args.latent_steps = 8
    model.model.forward.reset_mock()
    uc.execute(model, args)
    assert model.model.forward.call_count == forwards
    assert (migrated / "samples/000000/shared/ls5/hidden.pt").read_bytes() == old_bytes
    state = next(
        path
        for path in migrated.parent.iterdir()
        if path != migrated and (path / "state.json").is_file()
    )
    _, actual_hidden, _ = storage.assemble_shared_state(state, "000000", 8)
    fresh_args = SimpleNamespace(**vars(args))
    _, fresh_hidden, _ = InterventionContinuationRunner(
        model, fresh_args, evaluator
    ).compute_common_latent(item)
    assert torch.equal(actual_hidden, fresh_hidden)
