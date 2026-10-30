from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage, ResearchDataReader


def test_execution_reuses_shared_state_and_isolates_results(tmp_path):
    store = FileStorage(PathResolver(tmp_path))
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "question", "gold": "answer"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("answer", True, None)
    tracker = MagicMock()
    uc = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=store,
        tracker_port=tracker,
    )
    model = _build_mock_wrapper()
    args = SimpleNamespace(
        task="gsm8k",
        split="test",
        model_name="Qwen/model",
        latent_steps=15,
        latent_step_checkpoint_interval=10,
        intervention_conditions=["ple"],
        temperature=0.0,
        max_new_tokens=1,
        seed=42,
        no_progress=True,
    )
    uc.execute(model, args)
    state = next(store.base_dir.glob("intervention_continuation/*/*/*"))
    assert all(
        (state / "samples/000000/shared" / f"ls{s}" / "result.json").exists()
        for s in (0, 10, 15)
    )
    args.latent_steps = 20
    uc.execute(model, args)
    for step in (0, 10, 15, 20):
        kv, hidden, _ = store.assemble_shared_state(state, "000000", step)
        assert kv[0][0].shape[-2] == hidden.shape[0] == 3 + step
    args.temperature = 0.5
    uc.execute(model, args)
    assert (
        len(list((state / "samples/000000/results/ls20").glob("*/ple/result.json")))
        == 2
    )
    assert len(list(store.base_dir.glob("intervention_continuation/*/*/*"))) == 1
    run = next((state / "runs").iterdir())
    reader = ResearchDataReader(
        path_resolver=PathResolver(tmp_path), tensor_store=store
    )
    relative = str(run.relative_to(store.base_dir))
    _, hidden, _ = reader.load_intervention_state(relative, "000000")
    assert hidden.shape[0] in (18, 23)
    assert tracker.start_run.call_count == 3
    dataset.load.return_value = [{"question": "different", "gold": "answer"}]
    with pytest.raises(ValueError, match="sample identity"):
        uc.execute(model, args)


def test_later_segment_can_be_split_without_mutation(tmp_path):
    store = FileStorage(PathResolver(tmp_path))
    state = tmp_path / "state"
    original = (
        (torch.arange(23).reshape(1, 1, 23, 1), torch.arange(23).reshape(1, 1, 23, 1)),
    )
    from latentmas_reprop.domain.services.kv_cache import slice_past_kv

    hidden = torch.arange(23).reshape(23, 1)
    for start, end, left, right in [(0, 0, 0, 3), (0, 10, 3, 13), (10, 20, 13, 23)]:
        store.save_shared_segment(
            state,
            "000000",
            start,
            end,
            slice_past_kv(original, left, right),
            hidden[: 3 + end],
            {"prompt_length": 3},
        )
    kv15, h15, _ = store.assemble_shared_state(state, "000000", 15)
    store.save_shared_segment(
        state, "000000", 10, 15, slice_past_kv(kv15, 13, 18), h15, {"prompt_length": 3}
    )
    kv20, _, _ = store.assemble_shared_state(state, "000000", 20)
    assert torch.equal(kv20[0][0], original[0][0])


def test_incompatible_environment_allows_read_but_rejects_extension(tmp_path):
    from latentmas_reprop.application.intervention_continuation.identity import (
        ensure_state,
    )

    state = tmp_path / "state"
    identity = {"model": "test"}
    assert ensure_state(state, identity, {"device": "cpu"})
    assert not ensure_state(state, identity, {"device": "cuda"})
    with pytest.raises(ValueError, match="identity"):
        ensure_state(state, {"model": "other"}, {"device": "cpu"})


def test_identity_separates_state_and_condition_settings():
    from latentmas_reprop.application.intervention_continuation.identity import (
        canonical_hash,
        result_identity,
        state_identity,
    )

    model = _build_mock_wrapper()
    args = SimpleNamespace(
        model_name="Qwen/test",
        task="gsm8k",
        split="test",
        p1_prompt="{question}",
        latent_steps=10,
        seed=1,
        temperature=0.0,
        control_signal="A",
    )
    state1, _ = state_identity(model, args, "transformers")
    item = {"question": "Q", "gold": "1"}
    ple1 = result_identity(canonical_hash(state1), 10, "ple", item, args, model)
    plce1 = result_identity(canonical_hash(state1), 10, "plce", item, args, model)
    args.control_signal = "B"
    args.latent_steps = 20
    state2, _ = state_identity(model, args, "transformers")
    assert state1 == state2
    assert result_identity(canonical_hash(state2), 10, "ple", item, args, model) == ple1
    assert (
        result_identity(canonical_hash(state2), 10, "plce", item, args, model) != plce1
    )
    args.p1_prompt = "different {question}"
    state3, _ = state_identity(model, args, "transformers")
    assert state3 != state1


def test_partial_prefill_is_repaired_and_existing_segments_reused(tmp_path):
    store = FileStorage(PathResolver(tmp_path))
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "question", "gold": "answer"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("answer", True, None)
    uc = InterventionContinuationUseCase(
        dataset_port=dataset, evaluator_port=evaluator, storage=store
    )
    model = _build_mock_wrapper()
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/model",
        latent_steps=20,
        latent_step_checkpoint_interval=10,
        intervention_conditions=["ple"],
        temperature=0.0,
        max_new_tokens=1,
        seed=42,
        no_progress=True,
    )
    uc.execute(model, args)
    state = next(store.base_dir.glob("intervention_continuation/*/*/*"))
    shared = state / "samples/000000/shared"
    original20 = (shared / "ls20/kv.pt").read_bytes()
    args.latent_steps = 15
    model.model.forward = MagicMock(wraps=model.model.forward)
    uc.execute(model, args)
    assert (shared / "ls15/result.json").is_file()
    assert model.model.forward.call_count == 0
    assert (shared / "ls20/kv.pt").read_bytes() == original20
    (shared / "ls0/kv.pt").unlink()
    dataset.load.return_value = [{"question": "other", "gold": "answer"}]
    with pytest.raises(ValueError, match="sample identity"):
        uc.execute(model, args)
    assert not (shared / "ls0/kv.pt").exists()
    dataset.load.return_value = [{"question": "question", "gold": "answer"}]
    args.latent_steps = 20
    model.model.forward.reset_mock()
    uc.execute(model, args)
    assert (shared / "ls0/kv.pt").is_file()
    assert model.model.forward.call_count == 1
    assert (shared / "ls20/kv.pt").read_bytes() == original20


def test_stochastic_forward_does_not_commit_prefill(tmp_path):
    store = FileStorage(PathResolver(tmp_path))
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "question", "gold": "answer"}]
    model = _build_mock_wrapper()
    forward = model.model.forward

    def stochastic(*args, **kwargs):
        torch.rand(1)
        return forward(*args, **kwargs)

    model.model.forward = stochastic
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/model",
        latent_steps=1,
        intervention_conditions=["ple"],
        temperature=0.0,
        max_new_tokens=1,
        no_progress=True,
    )
    uc = InterventionContinuationUseCase(dataset_port=dataset, storage=store)
    with pytest.raises(ValueError, match="Stochastic forward"):
        uc.execute(model, args)
    assert not list(store.base_dir.glob("**/shared/ls0/result.json"))
    model.model.forward = forward
    with pytest.raises(ValueError, match="invalidated"):
        uc.execute(model, args)
