"""Flow and integration tests for the secret-digit receiver acquisition protocol."""

from types import SimpleNamespace

import pytest
import torch

from latentmas_reprop.application.receiver_acquisition_use_case import (
    ReceiverAcquisitionUseCase,
    aggregate_receiver_records,
)
from latentmas_reprop.domain.models import ReceiverAcquisitionRecord
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
from run.cli import parse_args


class _Tracker:
    def __init__(self):
        self.tags = None

    def start_run(self, *, tags, **_kwargs):
        self.tags = tags
        raise RuntimeError("stop after run creation")

    def log_params(self, _params):
        pass

    def log_dict(self, *_args, **_kwargs):
        pass

    def end_run(self, *_args, **_kwargs):
        pass


def _record(condition="own", mode="full", source_digit=2, source_prob=0.7):
    probs = {str(i): 0.01 for i in range(10)}
    probs[str(source_digit if source_digit is not None else 1)] = source_prob
    return ReceiverAcquisitionRecord(
        sample_id="secret_digit_0000",
        sample_index=0,
        sample_key="key",
        target_digit=1,
        source_sample_id="secret_digit_0001" if source_digit is not None else None,
        source_sample_index=1 if source_digit is not None else None,
        source_sample_key="source" if source_digit is not None else None,
        source_digit=source_digit,
        context_mode=mode,
        condition=condition,
        candidate_probabilities=probs,
        candidate_log_probabilities=dict.fromkeys(probs, -1.0),
        top_token_candidates=[],
        candidate_mass=sum(probs.values()),
        predicted_digit=source_digit if source_digit is not None else 1,
        source_probability=source_prob if source_digit is not None else None,
        source_log_probability=-0.3 if source_digit is not None else None,
        source_rank=1 if source_digit is not None else None,
        target_probability=probs["1"],
        target_log_probability=-1.0,
        target_rank=2,
        source_margin=0.6 if source_digit is not None else None,
        content_follow_correct=True if source_digit is not None else None,
        target_retained=source_digit is None,
        cache_present=condition != "drop",
        cache_sequence_length=8 if condition != "drop" else 0,
        cache_bytes=128 if condition != "drop" else 0,
        num_layers=1 if condition != "drop" else 0,
        cache_dtype="torch.float32" if condition != "drop" else None,
        original_full_seq_len=8 if condition != "drop" else 0,
        retained_tail_len=8 if condition != "drop" else 0,
        retained_tail_start_position=None,
        retained_original_position_end=8 if condition != "drop" else None,
        receiver_position_start=8 if condition != "drop" else 0,
        receiver_prompt_tokens=4,
        latency_sec=0.1,
        error=None,
        hidden_state_artifact_key=None,
        seed=42,
        model="model",
        task="secret_digit",
        latent_steps=2,
    )


def test_metrics_pair_source_probability_against_same_sample_drop():
    own = _record()
    drop = _record("drop", "none", None, 0.0)
    drop.candidate_probabilities["2"] = 0.2
    metrics = aggregate_receiver_records([own, drop], 1, 2, 1.0, 0.2)
    assert metrics.source_probability_delta_vs_drop["full/own"] == pytest.approx(0.5)
    assert metrics.n_records_successful == 2
    assert metrics.to_mlflow_metrics()["prob_delta/full/own_vs_drop"] == pytest.approx(
        0.5
    )
    assert "prob/drop/target_mean" in metrics.to_mlflow_metrics()
    assert all("//" not in key for key in metrics.to_mlflow_metrics())
    assert own.to_dict()["source_digit"] == 2


def test_cli_routes_canonical_acquisition_preset():
    args = parse_args(["--acquisition", "-c", "lm_q30.6_secret_digit"])
    assert args.acquisition
    assert args.carrier_modes == ["full", "prompt_only", "latent_only"]
    assert args.context_modes == args.carrier_modes
    assert args.acquisition_conditions == [
        "own",
        "cross",
        "drop",
        "drop_position_matched",
    ]
    assert args.probe_sender_latents


def test_cli_rejects_intervention_and_acquisition_together():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--intervention",
                "--acquisition",
                "-c",
                "lm_q30.6_secret_digit",
            ]
        )


def test_acquisition_mlflow_tags_are_strings():
    tracker = _Tracker()
    use_case = ReceiverAcquisitionUseCase(tracker_port=tracker)
    args = SimpleNamespace(
        method="latent_mas",
        task="secret_digit",
        use_vllm=False,
        latent_steps=4,
        tracking_experiment_name="latentmas_receiver_acquisition",
        acquisition_cross_policy="different_digit_shift_v1",
        max_samples=2,
        seed=42,
        acquisition_conditions=["own", "cross", "drop"],
        context_modes=["full", "latent_only"],
        model_name="model",
    )
    model = SimpleNamespace()
    with pytest.raises(RuntimeError, match="stop after run creation"):
        use_case.execute(model, args)
    assert tracker.tags is not None
    assert tracker.tags["seed"] == "42"
    assert tracker.tags["latent_steps"] == "4"


class _RolloutBackbone:
    def __call__(
        self, *, input_ids=None, inputs_embeds=None, attention_mask, **_kwargs
    ):
        batch = (input_ids if input_ids is not None else inputs_embeds).shape[0]
        hidden = (
            input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
            if input_ids is not None
            else inputs_embeds + 1
        )
        length = attention_mask.shape[-1]
        cache = ((torch.zeros(batch, 1, length, 1),) * 2,)
        return SimpleNamespace(past_key_values=cache, hidden_states=(hidden,))


def test_latent_rollout_captures_pre_and_post_realign_states():
    wrapper = object.__new__(ModelWrapper)
    wrapper.model = _RolloutBackbone()
    wrapper.device = torch.device("cpu")
    wrapper._apply_latent_realignment = lambda hidden, _model: hidden * 2
    result = wrapper.generate_latent_batch_with_states(
        torch.tensor([[2, 3]]), latent_steps=2
    )
    assert result.hidden_pre_realign.shape == (1, 2, 3)
    assert result.latent_post_realign.shape == (1, 2, 3)
    assert torch.equal(result.latent_post_realign, result.hidden_pre_realign * 2)
    assert not result.hidden_pre_realign.requires_grad


class _PositionBackbone:
    def __init__(self):
        self.position_ids = None

    def __call__(self, *, input_ids, position_ids, **_kwargs):
        self.position_ids = position_ids
        return SimpleNamespace(
            logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 20)
        )


def test_next_token_forward_preserves_requested_receiver_position_start():
    wrapper = object.__new__(ModelWrapper)
    wrapper.model = _PositionBackbone()
    wrapper.device = torch.device("cpu")
    wrapper.use_vllm = False
    cache = ((torch.zeros(1, 1, 4, 1), torch.zeros(1, 1, 4, 1)),)
    wrapper.forward_next_token_batch(
        torch.ones(1, 3, dtype=torch.long),
        past_key_values=cache,
        position_start=38,
    )
    assert wrapper.model.position_ids.tolist() == [[38, 39, 40]]
