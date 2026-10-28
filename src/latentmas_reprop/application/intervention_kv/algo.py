"""Cross-sample pairing and permutation algorithms for latent intervention."""

import random


def generate_cross_indices(
    n: int, policy: str = "derangement", seed: int = 42
) -> list[int]:
    """Generate permutation indices for cross-sample KV intervention.

    Guarantees result[i] != i for all i in range(n) (derangement property).

    Supported policies:
    - "derangement": rejection-sampled uniform random derangement (seed reproducible)
    - "shift_1": deterministic (i + 1) % n cyclic shift
    - "reverse": deterministic reversed indices with fixed point handling
    """
    if n < 2:
        raise ValueError(
            f"Cannot generate cross indices for n={n}: requires at least 2 samples"
        )

    if policy == "shift_1":
        return [(i + 1) % n for i in range(n)]

    if policy == "reverse":
        indices = list(reversed(range(n)))
        if n % 2 == 1:
            mid = n // 2
            indices[mid], indices[(mid + 1) % n] = indices[(mid + 1) % n], indices[mid]
        return indices

    if policy == "derangement":
        rng = random.Random(seed)
        for _ in range(10_000):
            perm = list(range(n))
            rng.shuffle(perm)
            if all(perm[i] != i for i in range(n)):
                return perm

        # Fallback: Early-Fisher derangement
        perm = [(i + 1) % n for i in range(n)]
        for i in range(n - 1, 0, -1):
            j = rng.randint(0, i - 1)
            perm[i], perm[j] = perm[j], perm[i]
        if all(perm[i] != i for i in range(n)):
            return perm
        return [(i + 1) % n for i in range(n)]

    raise ValueError(f"Unknown cross policy: {policy}")
