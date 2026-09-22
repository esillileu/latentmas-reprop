"""CPU-only tests for the secret-digit receiver acquisition protocol."""

import re
from types import SimpleNamespace

import pytest
import torch

from latentmas_reprop.application.receiver_acquisition_use_case import (
    ReceiverAcquisitionUseCase,
    aggregate_receiver_records,
    build_receiver_messages,
    generate_secret_digit_samples,
    pair_different_digit_sources,
    score_digit_logits,
    validate_digit_candidates,
)
from latentmas_reprop.application.sender_latent_probe import run_sender_latent_probe
from latentmas_reprop.domain.models import ReceiverAcquisitionRecord
from latentmas_reprop.domain.ports.model_port import LatentRolloutState
from latentmas_reprop.domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    retain_past_kv_prefix,
    truncate_past_kv,
)
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


def test_secret_digit_samples_are_deterministic_balanced_and_seeded():
    first = generate_secret_digit_samples(23, 42)
    again = generate_secret_digit_samples(23, 42)
    other = generate_secret_digit_samples(23, 43)
    assert first == again
    assert [sample.digit for sample in first] != [sample.digit for sample in other]
    counts = [sum(sample.digit == digit for sample in first) for digit in range(10)]
    assert max(counts) - min(counts) <= 1
    assert first[0].sample_id == "secret_digit_0000"
    assert len({sample.sample_key for sample in first}) == len(first)


@pytest.mark.parametrize("count", [1, 7, 10, 21])
def test_secret_digit_balance_for_sample_sizes(count):
    samples = generate_secret_digit_samples(count, 7)
    counts = [sum(sample.digit == digit for sample in samples) for digit in range(10)]
    assert max(counts) - min(counts) <= 1


def test_cross_sources_are_distinct_samples_and_digits():
    samples = generate_secret_digit_samples(25, 12)
    pairs = pair_different_digit_sources(samples)
    assert all(source != target for target, source in enumerate(pairs))
    assert all(
        samples[source].digit != samples[target].digit
        for target, source in enumerate(pairs)
    )


def test_cross_pairing_rejects_impossible_input():
    with pytest.raises(ValueError, match="different digits"):
        pair_different_digit_sources(generate_secret_digit_samples(1, 1))


def test_truncation_uses_clone_and_preserves_full_cache():
    full = ((torch.arange(12).reshape(1, 1, 6, 2), torch.ones(1, 1, 6, 2)),)
    original = full[0][0].clone()
    latent = truncate_past_kv(clone_past_kv(full), 2)
    assert get_past_kv_sequence_length(full) == 6
    assert get_past_kv_sequence_length(latent) == 2
    assert torch.equal(full[0][0], original)
    assert torch.equal(latent[0][0], original[..., -2:, :])


def test_prompt_only_uses_cache_prefix_without_mutating_full_cache():
    full = ((torch.arange(12).reshape(1, 1, 6, 2), torch.ones(1, 1, 6, 2)),)
    prompt = retain_past_kv_prefix(clone_past_kv(full), 4)
    assert get_past_kv_sequence_length(full) == 6
    assert get_past_kv_sequence_length(prompt) == 4
    assert torch.equal(prompt[0][0], full[0][0][..., :4, :])


class _DigitTokenizer:
    def __call__(self, text, **_kwargs):
        if text.startswith("prompt"):
            token_ids = [100, 101, 102]
            if text[-1].isdigit():
                token_ids.append(200 + int(text[-1]))
            return {"input_ids": token_ids}
        return {"input_ids": [200 + int(text)]}


class _TokenModel:
    tokenizer = _DigitTokenizer()

    def tokenize_text(self, text):
        return torch.tensor([[200 + int(text)]])


def test_candidate_validation_and_scoring_use_full_vocabulary_softmax():
    mapping = validate_digit_candidates(_TokenModel(), "prompt")
    logits = torch.zeros(1, 220)
    logits[0, 203] = 4
    logits[0, 207] = 2
    result = score_digit_logits(logits, mapping, source_digit=3, target_digit=7)
    assert result["predicted_digit"] == 3
    assert result["source_rank"] == 1
    assert result["target_rank"] == 2
    assert result["source_margin"] > 0
    assert result["candidate_mass"] < 1


def test_receiver_prompt_builder_accepts_no_secret_input():
    assert build_receiver_messages() == [
        {
            "role": "user",
            "content": (
                "Identify the secret digit from the sender's internal state. "
                "Your answer must have exactly this form: The number is <digit>"
            ),
        }
    ]


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


class _ProbeModel:
    def prepare_chat_batch(self, messages, add_generation_prompt=True):
        del add_generation_prompt
        digit = int(re.search(r"\d", messages[0][0]["content"]).group())
        ids = torch.tensor([[digit]])
        return [messages[0][0]["content"]], ids, torch.ones_like(ids), [[]]

    def generate_latent_batch_with_states(
        self, input_ids, attention_mask=None, *, latent_steps, past_key_values=None
    ):
        del attention_mask, past_key_values
        digit = int(input_ids.item())
        state = torch.nn.functional.one_hot(
            torch.tensor(digit), num_classes=10
        ).float()
        steps = state.reshape(1, 1, 10).repeat(1, latent_steps, 1)
        return LatentRolloutState(None, steps, steps)


def test_sender_probe_uses_template_disjoint_balanced_split():
    args = SimpleNamespace(
        probe_prompt_templates=2,
        probe_train_template_fraction=0.5,
        probe_epochs=1,
        latent_steps=2,
        seed=42,
    )
    result = run_sender_latent_probe(_ProbeModel(), args)
    assert set(result.split["train_template_indices"]).isdisjoint(
        result.split["test_template_indices"]
    )
    assert result.states["latent_post_realign"].shape == (20, 2, 10)
    assert result.summary["n_train_examples"] == 10
    assert result.summary["n_test_examples"] == 10
    assert {
        item["digit"] for item in result.metadata if item["split"] == "test"
    } == set(range(10))


class _RolloutBackbone:
    def __call__(self, *, input_ids=None, inputs_embeds=None, attention_mask, **_kwargs):
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
    assert torch.equal(
        result.latent_post_realign, result.hidden_pre_realign * 2
    )
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
