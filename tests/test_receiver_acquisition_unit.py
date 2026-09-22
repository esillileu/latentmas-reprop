"""Unit tests for secret-digit sampling, pairing, cache truncation, and token scoring."""

import pytest
import torch

from latentmas_reprop.application.receiver_acquisition_use_case import (
    build_receiver_messages,
    generate_secret_digit_samples,
    pair_different_digit_sources,
    score_digit_logits,
    validate_digit_candidates,
)
from latentmas_reprop.domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    retain_past_kv_prefix,
    truncate_past_kv,
)


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
