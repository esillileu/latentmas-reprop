"""Dataset generation, sampling, and pairing for receiver acquisition experiments."""

import random
from dataclasses import dataclass
from typing import Any

from ...domain.models import compute_sample_key


@dataclass(frozen=True)
class SecretDigitSample:
    """Problem sample for digit communication."""

    sample_id: str
    sample_index: int
    sample_key: str
    digit: int


@dataclass(frozen=True)
class SenderCacheBundle:
    """Collection of sender KV cache variants."""

    full: Any
    prompt_only: Any
    latent_only: Any
    prompt_len: int
    full_len: int
    build_latency_sec: float


def generate_secret_digit_samples(
    max_samples: int, seed: int
) -> list[SecretDigitSample]:
    """Generate deterministic, balanced secret digit problem samples."""
    if max_samples <= 0:
        raise ValueError("secret_digit acquisition requires max_samples > 0")
    digits = [index % 10 for index in range(max_samples)]
    random.Random(seed).shuffle(digits)
    return [
        SecretDigitSample(
            sample_id=f"secret_digit_{index:04d}",
            sample_index=index,
            sample_key=compute_sample_key(
                "secret_digit", str(seed), f"{index}:{digit}"
            ),
            digit=digit,
        )
        for index, digit in enumerate(digits)
    ]


def pair_different_digit_sources(samples: list[SecretDigitSample]) -> list[int]:
    """Pair each target sample with a different-digit source sample."""
    if len(samples) < 2 or len({sample.digit for sample in samples}) < 2:
        raise ValueError("cross acquisition requires at least two different digits")
    result: list[int] = []
    for index, target in enumerate(samples):
        source = next(
            (
                (index + offset) % len(samples)
                for offset in range(1, len(samples))
                if samples[(index + offset) % len(samples)].digit != target.digit
            ),
            None,
        )
        if source is None:
            raise ValueError("could not find a different-digit cross source")
        assert source != index and samples[source].digit != target.digit
        result.append(source)
    return result
