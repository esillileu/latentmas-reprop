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
