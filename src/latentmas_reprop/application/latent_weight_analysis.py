"""Readout-start distributions and injected vectors from recorded model weights."""

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from ..infrastructure.storage import load_tensor
from .intervention_continuation.identity import _tensor_digest


def load_embedding_weights(metadata, device):
    """Read only input/output matrices from the exact local model revision."""
    import json
    from pathlib import Path

    def cached(filename):
        return hf_hub_download(
            metadata["model_name"],
            filename,
            revision=metadata["model_revision"],
            local_files_only=True,
        )

    index = json.loads(Path(cached("model.safetensors.index.json")).read_text())[
        "weight_map"
    ]
    values = []
    for name in ("model.embed_tokens.weight", "lm_head.weight"):
        with safe_open(cached(index[name]), framework="pt", device="cpu") as stream:
            values.append(
                stream.get_tensor(name).to(
                    device=device,
                    dtype=getattr(torch, metadata["dtype"].split(".")[-1]),
                )
            )
    return values


def load_alignment(metadata, path, device):
    """Require the recorded transform, rather than substituting a new ridge fit."""
    value = load_tensor(path, map_location="cpu")
    matrix = None if value.get("identity", False) else value["matrix"]
    tensors = [("target_norm", value["target_norm"])]
    if matrix is not None:
        tensors.append(("matrix", matrix))
    if _tensor_digest(tensors) != metadata["state_identity"]["realignment_digest"]:
        raise ValueError("Alignment weights differ from the recorded transform")
    if (matrix is not None) != metadata["latent_space_realign"]:
        raise ValueError("Alignment mode differs from the recorded transform")
    return (matrix.to(device) if matrix is not None else None), value["target_norm"].to(
        device
    )


def injected_vectors(states, matrix, target_norm):
    """States at 0..S-1 predict the vectors injected at steps 1..S."""
    aligned = states.float() if matrix is None else states.float() @ matrix.float()
    return (
        aligned * (target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    ).to(states.dtype)


def distribution_features(hidden, head, markers):
    logits = F.linear(hidden.to(head.device, head.dtype), head).float()
    logs = logits.log_softmax(-1)
    p = logs.exp()
    return {
        "entropy": -(p * logs).sum().item(),
        "top_token_id": int(logits.argmax()),
        "top_probability": p.max().item(),
        **{
            f"{name}_{field}": values[token].item()
            for name, token in zip(("think", "endthink"), markers, strict=True)
            for field, values in (("logit", logits), ("probability", p))
        },
    }


def embedding_comparison(vectors, embedding, chunk_size=2048):
    """Exact nearest cosine against the vocabulary, without a full float copy."""
    unit = F.normalize(vectors.float(), dim=-1)
    best = torch.full((len(vectors),), -float("inf"), device=vectors.device)
    nearest = torch.zeros(len(vectors), dtype=torch.long, device=vectors.device)
    norms = []
    centroid = torch.zeros(embedding.shape[1], device=vectors.device)
    for start in range(0, len(embedding), chunk_size):
        chunk = embedding[start : start + chunk_size].float()
        norms.append(chunk.norm(dim=-1))
        centroid += chunk.sum(dim=0)
        scores, ids = (unit @ F.normalize(chunk, dim=-1).T).max(dim=-1)
        update = scores > best
        best = torch.where(update, scores, best)
        nearest = torch.where(update, ids + start, nearest)
    norms = torch.cat(norms)
    centroid /= len(embedding)
    vector_norms = vectors.float().norm(dim=-1)
    return [
        {
            "injected_norm": vector_norms[i].item(),
            "embedding_norm_percentile": (norms <= vector_norms[i])
            .float()
            .mean()
            .item(),
            "embedding_norm_mean": norms.mean().item(),
            "embedding_norm_std": norms.std(unbiased=False).item(),
            "nearest_embedding_cosine": best[i].item(),
            "nearest_embedding_token_id": int(nearest[i]),
            "embedding_centroid_cosine": F.cosine_similarity(
                vectors[i].float(), centroid, dim=0
            ).item(),
        }
        for i in range(len(vectors))
    ]
