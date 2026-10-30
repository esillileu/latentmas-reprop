"""Integration tests for incremental execution and PE."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage


def test_pe_uses_prefill_and_resume_matches_continuous(tmp_path):
    wrapper = _build_mock_wrapper()
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=20,
        max_new_tokens=1,
        temperature=0.7,
        top_p=1.0,
        seed=42,
        max_samples=1,
        no_progress=True,
        intervention_conditions=["pe"],
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    storage = FileStorage(PathResolver(tmp_path / "resumed"))
    uc = InterventionContinuationUseCase(
        dataset_port=dataset, storage=storage, evaluator_port=evaluator
    )
    _, pe = uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 1
    assert pe[0]["condition"] == "pe"
    assert list(
        storage.base_dir.glob(
            "intervention_continuation/*/*/*/samples/000000/results/ls0/*/pe/result.json"
        )
    )
    args.intervention_conditions = ["ple"]
    wrapper.model.forward.reset_mock()
    uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 20
    args.latent_steps = 40
    wrapper.model.forward.reset_mock()
    _, resumed = uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 20
    fresh = InterventionContinuationUseCase(
        dataset_port=dataset,
        storage=FileStorage(PathResolver(tmp_path / "fresh")),
        evaluator_port=evaluator,
    )
    _, continuous = fresh.execute(wrapper, args)
    assert resumed[0]["generated_token_ids"] == continuous[0]["generated_token_ids"]
    wrapper.model.forward.reset_mock()
    uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0


def test_prefetch_overlaps_decode_and_propagates_errors(tmp_path, monkeypatch):
    import threading

    import pytest

    from latentmas_reprop.application.intervention_continuation.runner import (
        InterventionContinuationRunner,
    )

    wrapper = _build_mock_wrapper()
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=0,
        max_new_tokens=1,
        temperature=0.7,
        top_p=1.0,
        seed=42,
        max_samples=2,
        no_progress=True,
        intervention_conditions=["pe"],
    )
    dataset = MagicMock()
    dataset.load.return_value = [
        {"question": "Q1", "gold": "1"},
        {"question": "Q2", "gold": "1"},
    ]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    storage = FileStorage(PathResolver(tmp_path))
    uc = InterventionContinuationUseCase(
        dataset_port=dataset, storage=storage, evaluator_port=evaluator
    )
    uc.execute(wrapper, args)
    args.seed = 43
    started = threading.Event()
    decoding = threading.Event()
    original_assemble = storage.assemble_shared_state
    original_decode = InterventionContinuationRunner.execute_ple

    def assemble(state, sample, step):
        if (
            sample == "000001"
            and threading.current_thread() is not threading.main_thread()
        ):
            started.set()
            assert decoding.wait(10)
        return original_assemble(state, sample, step)

    def decode(self, *a):
        assert started.wait(10)
        decoding.set()
        return original_decode(self, *a)

    monkeypatch.setattr(storage, "assemble_shared_state", assemble)
    monkeypatch.setattr(InterventionContinuationRunner, "execute_ple", decode)
    _, pipelined = uc.execute(wrapper, args)
    from latentmas_reprop.application.intervention_continuation import execution

    original_prepare = execution._prepare_sample
    monkeypatch.setattr(execution, "_prepare_sample", lambda *a: {})
    sequential_storage = FileStorage(PathResolver(tmp_path / "sequential"))
    sequential = InterventionContinuationUseCase(
        dataset_port=dataset, storage=sequential_storage, evaluator_port=evaluator
    )
    _, sequential_records = sequential.execute(wrapper, args)
    assert pipelined == sequential_records
    import torch

    for record in pipelined:
        sample = f"{record['sample_index']:06d}"
        pattern = f"intervention_continuation/*/*/*/samples/{sample}/results/ls0/*/pe/hidden.pt"
        pipelined_files = list(storage.base_dir.glob(pattern))
        sequential_file = next(sequential_storage.base_dir.glob(pattern))
        assert any(
            torch.equal(
                torch.load(path, weights_only=True),
                torch.load(sequential_file, weights_only=True),
            )
            for path in pipelined_files
        )
    monkeypatch.setattr(execution, "_prepare_sample", original_prepare)

    args.seed = 44
    calls = []

    def fail(state, sample, step):
        if (
            sample == "000001"
            and threading.current_thread() is not threading.main_thread()
        ):
            raise OSError("prefetch failed")
        return original_assemble(state, sample, step)

    def record_decode(self, *a):
        calls.append(a[0]["question"])
        return original_decode(self, *a)

    monkeypatch.setattr(storage, "assemble_shared_state", fail)
    monkeypatch.setattr(InterventionContinuationRunner, "execute_ple", record_decode)
    with pytest.raises(OSError, match="prefetch failed"):
        uc.execute(wrapper, args)
    assert calls == ["Q1"]


def test_condition_aliases_and_intervention_content():
    from latentmas_reprop.application.intervention_continuation.identity import (
        intervention_messages,
        resolve_conditions,
    )

    args = SimpleNamespace(model_name="Qwen/mock", task="gsm8k")
    assert resolve_conditions(args) == [
        "pe",
        "ple",
        "plce",
        "rqlre",
        "rqlqe",
        "plpe",
        "rqrqe",
    ]
    args.p1_variant = "question_only"
    assert resolve_conditions(args) == ["pe", "ple"]
    item = {"question": "Distinct question text"}
    role = intervention_messages("rqlre", item, args)[0]["content"]
    question = intervention_messages("rqlqe", item, args)[0]["content"]
    assert "Judger" in role and item["question"] not in role
    assert item["question"] in question and "Judger" not in question
