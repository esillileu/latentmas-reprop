"""Integration tests for InterventionUseCase covering all conditions and resume behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.domain.models import compute_sample_key
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    load_json,
)


def _make_mock_model():
    model = MagicMock()
    model.model_name = "test_model"
    model.device = torch.device("cpu")
    model.model = MagicMock()
    model.model.dtype = "float32"
    return model


class _Dataset:
    def __init__(self, samples):
        self._samples = samples

    def load(self, task, split="test", **kwargs):
        return list(self._samples)


def test_intervention_use_case_execute_all_conditions(tmp_path):
    """Integration test: InterventionUseCase.execute() covering own, cross, drop, zero."""
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)

    samples = [
        {"question": "What is 2 + 2?", "solution": "4", "gold": "4"},
        {"question": "What is 3 + 3?", "solution": "6", "gold": "6"},
    ]

    dataset_port = _Dataset(samples)
    evaluator_port = MagicMock()
    evaluator_port.evaluate.return_value = True

    tracker = MagicMock()
    tracker.active_run_id = None

    uc = InterventionUseCase(
        cache_port=cache,
        tracker_port=tracker,
        dataset_port=dataset_port,
        evaluator_port=evaluator_port,
    )

    args = SimpleNamespace(
        model_name="test_model",
        task="gsm8k",
        method="latent_mas",
        split="test",
        prompt="sequential",
        seed=42,
        max_samples=2,
        latent_steps=5,
        max_new_tokens=32,
        temperature=0.6,
        top_p=0.95,
        intervention_conditions="own,cross,drop,zero",
        cross_pairing_policy="shift_1",
        save_raw_cache=True,
        save_hidden_states=False,
        tracking_experiment_name="test_exp",
        run_id="integ_run_01",
    )

    model = _make_mock_model()

    dummy_kv = ((torch.ones((1, 2, 4, 8)), torch.ones((1, 2, 4, 8))),)

    with (
        patch(
            "latentmas_reprop.application.intervention_use_case.LatentMASMethod.build_latent_contexts",
            return_value=(dummy_kv, [[{"trace": "planner"}]]),
        ),
        patch(
            "latentmas_reprop.application.intervention_use_case.LatentMASMethod.decode_with_context",
            return_value=[
                {
                    "prediction": "4",
                    "raw_prediction": "The answer is 4",
                    "error_msg": None,
                    "generated_tokens": 5,
                    "judger_trace": {"step": 1},
                }
            ],
        ) as mock_decode,
    ):
        _metrics, records = uc.execute(model=model, args=args)

        # 2 samples * 4 conditions = 8 records
        assert len(records) == 8
        assert mock_decode.call_count == 8

        # Conditions verified
        conditions_found = {r.condition for r in records}
        assert conditions_found == {"own", "cross", "drop", "zero"}

        # Verify cross pairing logic: shift_1 pairs idx 0 with 1, and idx 1 with 0
        key_0 = compute_sample_key("gsm8k", "test", samples[0]["question"])
        key_1 = compute_sample_key("gsm8k", "test", samples[1]["question"])

        cross_0 = next(
            r for r in records if r.sample_index == 0 and r.condition == "cross"
        )
        assert cross_0.source_sample_index == 1
        assert cross_0.sample_key == key_0
        assert cross_0.source_sample_key == key_1

        cross_1 = next(
            r for r in records if r.sample_index == 1 and r.condition == "cross"
        )
        assert cross_1.source_sample_index == 0
        assert cross_1.sample_key == key_1
        assert cross_1.source_sample_key == key_0

        # Verify disk storage integrity
        exp_dir = resolver.get_experiment_dir("integ_run_01")
        assert (exp_dir / "config.json").is_file()

        for s_idx, key in enumerate([key_0, key_1]):
            for cond in ["own", "cross", "drop", "zero"]:
                cond_dir = resolver.get_condition_dir("integ_run_01", key, cond)
                assert is_condition_complete(cond_dir)
                data = load_json(cond_dir / "result.json")
                assert data["sample_index"] == s_idx
                assert data["condition"] == cond

        # Now test Resume: Re-running with same explicit run_id must skip already completed decodings
        mock_decode.reset_mock()
        _metrics2, records2 = uc.execute(model=model, args=args)
        assert len(records2) == 8
        # All conditions were complete on disk, so decode should NOT have been called!
        assert mock_decode.call_count == 0
