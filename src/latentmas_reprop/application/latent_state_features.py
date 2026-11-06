"""Geometry and raw next-token distributions of saved shared trajectories."""

import torch
import torch.nn.functional as F


def hidden_features(hidden, prompt_hidden=None):
    """Index latent outputs at steps 1+, with prompt final state at step zero."""
    if hidden.ndim != 2 or not hidden.shape[1] or not torch.isfinite(hidden).all():
        raise ValueError("Hidden states must be a finite [steps, hidden_dim] matrix")
    states = {index + 1: state.float() for index, state in enumerate(hidden)}
    if prompt_hidden is not None:
        if (
            prompt_hidden.ndim != 2
            or prompt_hidden.shape[1] != hidden.shape[1]
            or not len(prompt_hidden)
            or not torch.isfinite(prompt_hidden[-1]).all()
        ):
            raise ValueError("Invalid prompt hidden state")
        states[0] = prompt_hidden[-1].float()
    rows = {}
    for step, state in sorted(states.items()):
        previous = states.get(step - 1)
        origin = states.get(0)
        row = {
            "step": step,
            "l2_norm": state.norm().item(),
            "hidden_mean": state.mean().item(),
            "hidden_std": state.std(unbiased=False).item(),
            "hidden_max_abs": state.abs().max().item(),
            "adjacent_cosine": None,
            "adjacent_l2": None,
            "direction_cosine": None,
            "distance_0": None,
            "cosine_0": None,
        }
        if origin is not None:
            row.update(
                distance_0=(state - origin).norm().item(),
                cosine_0=F.cosine_similarity(state, origin, dim=0).item(),
            )
        if previous is not None:
            row.update(
                adjacent_cosine=F.cosine_similarity(previous, state, dim=0).item(),
                adjacent_l2=(state - previous).norm().item(),
            )
        if previous is not None and step - 2 in states:
            row["direction_cosine"] = F.cosine_similarity(
                previous - states[step - 2], state - previous, dim=0
            ).item()
        rows[step] = row
    return rows


def logit_features(logits, markers=None, tokenizer=None):
    """Step 0 logits are not saved. Initial-logit comparisons use step 1."""
    if logits.ndim != 2 or not logits.shape[1] or not torch.isfinite(logits).all():
        raise ValueError("Logits must be a finite [steps, vocab_size] matrix")
    logp = logits.float().log_softmax(-1)
    probabilities = logp.exp()
    rows = {}
    for index, (logs, probs) in enumerate(zip(logp, probabilities, strict=True)):
        top = probs.topk(min(10, len(probs)))
        row = {
            "step": index + 1,
            "entropy": -(probs * logs).sum().item(),
            "top_token_ids": top.indices.tolist(),
            "top_probabilities": top.values.tolist(),
            "top_tokens": tokenizer.convert_ids_to_tokens(top.indices.tolist())
            if tokenizer is not None
            else None,
            "top1_probability": top.values[0].item(),
            "top10_probability_mass": top.values.sum().item(),
            "js_divergence": None,
            "kl_current_previous": None,
            "js_from_step1": None,
            "logit_reference_step": 1,
            "think_logit": None,
            "endthink_logit": None,
            "think_probability": None,
            "endthink_probability": None,
        }
        for field, reference in (("js_divergence", index - 1), ("js_from_step1", 0)):
            if reference < 0:
                continue
            logm = torch.logaddexp(logs, logp[reference]) - torch.log(
                logs.new_tensor(2.0)
            )
            row[field] = max(
                0.0,
                (
                    0.5
                    * (
                        (probs * (logs - logm)).sum()
                        + (probabilities[reference] * (logp[reference] - logm)).sum()
                    )
                ).item(),
            )
        if index:
            row["kl_current_previous"] = max(
                0.0, (probs * (logs - logp[index - 1])).sum().item()
            )
        if markers is not None:
            if any(
                type(token) is not int or not 0 <= token < len(probs)
                for token in markers
            ):
                raise ValueError("Thinking token IDs outside saved vocabulary")
            for label, token in zip(("think", "endthink"), markers, strict=True):
                row[f"{label}_logit"] = logits[index, token].item()
                row[f"{label}_probability"] = probs[token].item()
        rows[index + 1] = row
    return rows


def fit_pca(hidden, weights=None):
    """Fit a shared mean/basis; optional point weights sum to one."""
    values = hidden.float()
    if (
        values.ndim != 2
        or not len(values)
        or not values.shape[1]
        or not torch.isfinite(values).all()
    ):
        raise ValueError("PCA requires a nonempty finite state matrix")
    if weights is None:
        weights = torch.ones(len(values), dtype=values.dtype, device=values.device)
    if (
        weights.shape != (len(values),)
        or not torch.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("PCA weights must be positive and match states")
    weights = weights / weights.sum()
    mean = (values * weights[:, None]).sum(dim=0)
    matrix = (values - mean) * weights.sqrt()[:, None]
    _, singular, axes = torch.linalg.svd(matrix, full_matrices=False)
    tolerance = max(matrix.shape) * torch.finfo(matrix.dtype).eps * singular[0]
    nonzero = singular[singular > tolerance]
    rank = 0.0
    variance = torch.zeros_like(singular)
    if len(nonzero):
        probabilities = nonzero / nonzero.sum()
        rank = (-(probabilities * probabilities.log()).sum()).exp().item()
        variance = singular.square() / singular.square().sum()
    components = values.new_zeros((2, values.shape[1]))
    dimensions = min(2, len(axes))
    components[:dimensions] = axes[:dimensions]
    for component in components:
        if component[component.abs().argmax()] < 0:
            component *= -1
    explained = values.new_zeros(2)
    explained[:dimensions] = variance[:dimensions]
    return {
        "mean": mean,
        "components": components,
        "explained_variance_ratio": explained,
        "effective_rank": rank,
        "numerical_rank": len(nonzero),
        "n_components": dimensions,
    }


def trajectory_geometry(hidden, steps):
    """Centered local PCA, sharing the same fitting primitive as common PCA."""
    basis = fit_pca(hidden)
    coordinates = (hidden.float() - basis["mean"]) @ basis["components"].T
    rows = [
        {
            "step": step,
            "pca_1": coordinates[i, 0].item(),
            "pca_2": coordinates[i, 1].item(),
        }
        for i, step in enumerate(steps)
    ]
    return rows, {
        "effective_rank": basis["effective_rank"],
        "numerical_rank": basis["numerical_rank"],
        "pca_explained_variance": basis["explained_variance_ratio"].tolist(),
        "pca_centered": True,
        "rank_definition": "exp(entropy(normalized nonzero singular values of centered states))",
    }
