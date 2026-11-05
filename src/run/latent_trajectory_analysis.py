"""Analyze saved hidden states and raw vocabulary logits without model loading."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from latentmas_reprop.infrastructure.storage import save_json_atomic


def trajectory_metrics(data):
    hidden = data["hidden"].float()
    logits = data["logits"].float()
    logp = logits.log_softmax(-1)
    probabilities = logp.exp()
    delta = hidden[1:] - hidden[:-1]
    rows = []
    for index, step in enumerate(data["steps"].tolist()):
        top = probabilities[index].topk(min(10, logits.shape[-1]))
        row = {
            "step": step,
            "l2_norm": hidden[index].norm().item(),
            "entropy": -(probabilities[index] * logp[index]).sum().item(),
            "top_token_ids": top.indices.tolist(),
            "top_probabilities": top.values.tolist(),
            "adjacent_cosine": None,
            "adjacent_l2": None,
            "direction_cosine": None,
            "js_divergence": None,
        }
        if index:
            p, q = probabilities[index - 1], probabilities[index]
            logm = ((p + q) / 2).clamp_min(torch.finfo(p.dtype).tiny).log()
            row.update(
                adjacent_cosine=F.cosine_similarity(
                    hidden[index - 1], hidden[index], dim=0
                ).item(),
                adjacent_l2=delta[index - 1].norm().item(),
                js_divergence=(
                    0.5
                    * (
                        (p * (logp[index - 1] - logm)).sum()
                        + (q * (logp[index] - logm)).sum()
                    )
                ).item(),
            )
        if index >= 2:
            row["direction_cosine"] = F.cosine_similarity(
                delta[index - 2], delta[index - 1], dim=0
            ).item()
        rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    for path in sorted(args.directory.glob("samples/*/trajectories/*.pt")):
        rows = trajectory_metrics(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        save_json_atomic(path.with_suffix(".analysis.json"), rows)


if __name__ == "__main__":
    main()
