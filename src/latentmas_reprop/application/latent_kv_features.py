"""Layer/component structure of saved KV, using sequence-position vectors."""

import torch

RANK_FIELDS = (
    "numerical_rank",
    "effective_rank",
    "top1_energy_fraction",
    "rank_positions",
)


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return (
        (torch.dot(left, right) / denominator).clamp(-1, 1).item()
        if denominator > 0
        else None
    )


def kv_component_features(tensor, prompt_length, latent_steps):
    """Flatten heads/batch per position. Stored K is not de-rotated."""
    if (
        type(prompt_length) is not int
        or type(latent_steps) is not int
        or min(prompt_length, latent_steps) < 0
    ):
        raise ValueError("KV prompt/latent lengths must be nonnegative integers")
    if (
        tensor.ndim < 2
        or tensor.shape[-2] != prompt_length + latent_steps
        or not tensor.shape[-1]
    ):
        raise ValueError("KV sequence shape differs from prompt + latent steps")
    values = tensor.movedim(-2, 0).flatten(1).float()
    if not values.shape[1]:
        raise ValueError("KV vectors must be nonempty")
    finite = torch.isfinite(values).all(dim=1)
    norms = values.norm(dim=1)
    prompt_indices = torch.where(finite[:prompt_length] & (norms[:prompt_length] > 0))[
        0
    ]
    prompt = values[prompt_indices]
    normalized_prompt = (
        prompt / prompt.norm(dim=1, keepdim=True) if len(prompt) else None
    )
    centroid = prompt.mean(dim=0) if len(prompt) else None
    latent = values[prompt_length:][finite[prompt_length:]]
    summary = {
        "dtype": str(tensor.dtype),
        "vector_width": values.shape[1],
        "prompt_positions": prompt_length,
        "latent_positions": latent_steps,
        "valid_prompt_positions": finite[:prompt_length].sum().item(),
        "valid_latent_positions": len(latent),
        "rank_centered": False,
    }
    for name, selected in (
        ("prompt", values[:prompt_length][finite[:prompt_length]]),
        ("latent", latent),
    ):
        summary.update(
            {f"{name}_{key}": value for key, value in rank_features(selected).items()}
        )
        summary.update(
            {
                f"{name}_mean": selected.mean().item() if len(selected) else None,
                f"{name}_std": selected.std(unbiased=False).item()
                if len(selected)
                else None,
                f"{name}_norm_mean": selected.norm(dim=1).mean().item()
                if len(selected)
                else None,
            }
        )
    width = min(prompt_length, latent_steps)
    matched_prompt = values[prompt_length - width : prompt_length][
        finite[prompt_length - width : prompt_length]
    ]
    matched_latent = values[prompt_length : prompt_length + width][
        finite[prompt_length : prompt_length + width]
    ]
    summary["matched_window_positions"] = width
    for name, selected, full_length in (
        ("prompt", matched_prompt, prompt_length),
        ("latent", matched_latent, latent_steps),
    ):
        metrics = (
            {key: summary[f"{name}_{key}"] for key in RANK_FIELDS}
            if full_length == width
            else rank_features(selected)
        )
        summary.update(
            {f"matched_{name}_{key}": value for key, value in metrics.items()}
        )
    rows = []
    for position in range(len(values)):
        relative = position - prompt_length + 1
        region = "prompt" if position < prompt_length else "latent"
        row = {
            "step": relative if relative >= 0 else None,
            "relative_position": relative,
            "region": region,
            "region_position": position if region == "prompt" else relative,
            "cache_position": position,
            "status": "complete" if finite[position] else "invalid",
            "norm": None,
            "adjacent_cosine": None,
            "adjacent_l2": None,
            "prompt_last_cosine": None,
            "prompt_centroid_cosine": None,
            "nearest_prompt_cosine": None,
            "nearest_prompt_position": None,
        }
        if finite[position]:
            vector = values[position]
            row["norm"] = norms[position].item()
            if position > 0 and finite[position - 1]:
                row.update(
                    adjacent_cosine=cosine(vector, values[position - 1]),
                    adjacent_l2=(vector - values[position - 1]).norm().item(),
                )
            if prompt_length and finite[prompt_length - 1]:
                row["prompt_last_cosine"] = cosine(vector, values[prompt_length - 1])
            if centroid is not None:
                row["prompt_centroid_cosine"] = cosine(vector, centroid)
            if normalized_prompt is not None and norms[position] > 0:
                similarities = normalized_prompt @ (vector / norms[position])
                if position < prompt_length:
                    similarities[prompt_indices == position] = -torch.inf
                nearest = similarities.max(dim=0)
                if torch.isfinite(nearest.values):
                    row.update(
                        nearest_prompt_cosine=nearest.values.clamp(-1, 1).item(),
                        nearest_prompt_position=prompt_indices[nearest.indices].item(),
                    )
        rows.append(row)
    summary["status"] = "complete" if finite.all() else "partial"
    return summary, rows


def rank_features(matrix):
    """The same uncentered spectrum definition for both cache regions."""
    singular = torch.linalg.svdvals(matrix) if len(matrix) else torch.tensor([])
    active = (
        singular[
            singular > max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
        ]
        if len(singular)
        else singular
    )
    probabilities = active / active.sum() if len(active) else active
    rank = (
        (-(probabilities * probabilities.log()).sum()).exp().item()
        if len(active)
        else 0.0
    )
    energy = singular.square().sum()
    return {
        "numerical_rank": len(active),
        "effective_rank": rank if len(matrix) else None,
        "top1_energy_fraction": (singular[0].square() / energy).item()
        if energy > 0
        else None,
        "rank_positions": len(matrix),
    }
