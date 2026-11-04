"""Compare decoder outputs."""

from typing import Any

import numpy as np
import torch


def compare_equivalence(
    base_ids: list[int],
    base_hiddens: list[torch.Tensor],
    variant_ids: list[int],
    variant_hiddens: list[torch.Tensor],
) -> dict[str, Any]:
    """Check exact token id match and hidden state distance."""
    ids_match = base_ids == variant_ids
    num_common = min(len(base_hiddens), len(variant_hiddens))

    max_hidden_diff = 0.0
    mean_hidden_diff = 0.0
    if num_common > 0:
        diffs = []
        for h1, h2 in zip(
            base_hiddens[:num_common], variant_hiddens[:num_common], strict=False
        ):
            diff = torch.max(torch.abs(h1 - h2)).item()
            diffs.append(diff)
        max_hidden_diff = float(np.max(diffs))
        mean_hidden_diff = float(np.mean(diffs))

    is_equivalent = ids_match and (max_hidden_diff < 1e-3)

    return {
        "ids_exact_match": ids_match,
        "token_len_diff": len(variant_ids) - len(base_ids),
        "max_hidden_diff": max_hidden_diff,
        "mean_hidden_diff": mean_hidden_diff,
        "is_equivalent": is_equivalent,
    }
