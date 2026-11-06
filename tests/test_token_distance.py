"""Exact token edit distances, including bit boundaries and empty sequences."""

import random

from latentmas_reprop.domain.services.token_distance import token_levenshtein


def reference_distance(left, right):
    row = list(range(len(right) + 1))
    for i, token in enumerate(left, 1):
        following = [i]
        for j, other in enumerate(right, 1):
            following.append(
                min(following[-1] + 1, row[j] + 1, row[j - 1] + (token != other))
            )
        row = following
    return row[-1]


def test_distance_matches_dynamic_programming():
    rng = random.Random(42)
    for left_length in (0, 1, 2, 31, 64, 130):
        for right_length in (0, 1, 4, 32, 65, 140):
            left = [rng.randrange(8) for _ in range(left_length)]
            right = [rng.randrange(8) for _ in range(right_length)]
            expected = reference_distance(left, right)
            assert token_levenshtein(left, right) == expected
            assert token_levenshtein(right, left) == expected
    assert token_levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert token_levenshtein([1, 2, 3], [1, 4, 3]) == 1
