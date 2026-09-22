"""Candidate digit token validation and logit scoring utilities."""

import hashlib
from typing import Any

import torch

from ...domain.services.prompts.acquisition import (
    CANDIDATE_DIGITS,
    RECEIVER_ANSWER_PREFIX,
    RECEIVER_PROMPT_TEMPLATE_VERSION,
)
from ...infrastructure.models.model_wrapper import ModelWrapper


def validate_digit_candidates(
    model: ModelWrapper, rendered_prompt: str
) -> dict[str, Any]:
    """Validate that all candidate digits are single tokens in the scoring prompt context."""
    scoring_prompt = rendered_prompt + RECEIVER_ANSWER_PREFIX
    prefix = model.tokenizer(scoring_prompt, add_special_tokens=False)["input_ids"]
    mapping: dict[str, Any] = {}
    token_ids: list[int] = []
    for candidate in CANDIDATE_DIGITS:
        standalone = model.tokenize_text(candidate).detach().cpu().reshape(-1).tolist()
        contextual = model.tokenizer(
            scoring_prompt + candidate, add_special_tokens=False
        )["input_ids"]
        prefix_ok = contextual[: len(prefix)] == prefix
        suffix = contextual[len(prefix) :] if prefix_ok else []
        if len(standalone) != 1:
            raise ValueError(
                f"candidate {candidate!r} is not a standalone single token"
            )
        if not prefix_ok or len(suffix) != 1:
            raise ValueError(
                f"candidate {candidate!r} is not a contextual single-token suffix"
            )
        token_ids.append(int(suffix[0]))
        mapping[candidate] = {
            "standalone_token_id": int(standalone[0]),
            "contextual_token_id": int(suffix[0]),
            "prefix_unchanged": True,
        }
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("digit candidates do not map to unique contextual token IDs")
    variants: dict[str, Any] = {}
    for digit in CANDIDATE_DIGITS:
        variants[digit] = {}
        for label, candidate in (("plain", digit), ("space_prefixed", f" {digit}")):
            standalone = model.tokenizer(candidate, add_special_tokens=False)[
                "input_ids"
            ]
            contextual = model.tokenizer(
                scoring_prompt + candidate, add_special_tokens=False
            )["input_ids"]
            prefix_ok = contextual[: len(prefix)] == prefix
            variants[digit][label] = {
                "text": candidate,
                "standalone_token_ids": [int(token_id) for token_id in standalone],
                "contextual_token_ids": [
                    int(token_id) for token_id in contextual[len(prefix) :]
                ]
                if prefix_ok
                else [],
                "prefix_unchanged": prefix_ok,
            }
    return {
        "candidates": mapping,
        "candidate_variants": variants,
        "answer_prefix": RECEIVER_ANSWER_PREFIX,
        "mapping_unique": True,
        "rendered_receiver_prompt_hash": hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        "template_version": RECEIVER_PROMPT_TEMPLATE_VERSION,
    }


def score_digit_logits(
    logits: torch.Tensor,
    token_mapping: dict[str, Any],
    source_digit: int | None,
    target_digit: int,
) -> dict[str, Any]:
    """Score model output logits over the allowed candidate digits."""
    log_probs = torch.log_softmax(logits[0].float(), dim=-1)
    candidate_logs = {
        digit: float(log_probs[data["contextual_token_id"]].item())
        for digit, data in token_mapping["candidates"].items()
    }
    probabilities = {
        digit: float(torch.exp(torch.tensor(value)).item())
        for digit, value in candidate_logs.items()
    }
    ordered = sorted(probabilities, key=probabilities.get, reverse=True)
    predicted = int(ordered[0])

    def rank(digit: int) -> int:
        return ordered.index(str(digit)) + 1

    source_probability = (
        probabilities[str(source_digit)] if source_digit is not None else None
    )
    source_log_probability = (
        candidate_logs[str(source_digit)] if source_digit is not None else None
    )
    margin = None
    if source_digit is not None:
        other = max(
            value
            for digit, value in probabilities.items()
            if digit != str(source_digit)
        )
        margin = source_probability - other
    return {
        "candidate_probabilities": probabilities,
        "candidate_log_probabilities": candidate_logs,
        "candidate_mass": sum(probabilities.values()),
        "predicted_digit": predicted,
        "source_probability": source_probability,
        "source_log_probability": source_log_probability,
        "source_rank": rank(source_digit) if source_digit is not None else None,
        "target_probability": probabilities[str(target_digit)],
        "target_log_probability": candidate_logs[str(target_digit)],
        "target_rank": rank(target_digit),
        "source_margin": margin,
        "content_follow_correct": predicted == source_digit
        if source_digit is not None
        else None,
        "target_retained": predicted == target_digit,
    }
