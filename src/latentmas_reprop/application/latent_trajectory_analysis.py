"""Analyze saved hidden states and raw vocabulary logits without model loading."""

import re

import torch
import torch.nn.functional as F

from ..domain.services.token_distance import token_levenshtein
from ..infrastructure.storage import load_json, save_json_atomic


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


def analyze_trajectories(directory):
    for path in sorted(directory.glob("samples/*/trajectories/*.pt")):
        rows = trajectory_metrics(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        save_json_atomic(path.with_suffix(".analysis.json"), rows)


def readout_text(record, tokenizer):
    """Exclude terminal EOS, normalize whitespace, and retain symbols and tags."""
    ids = record["generated_token_ids"]
    length = record["content_length"]
    if not 0 <= length <= len(ids):
        raise ValueError("Invalid continuation content_length")
    text = tokenizer.decode(
        ids[:length], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return " ".join(text.split())


def adjacent_readout_metrics(sample_dir, metadata, tokenizer):
    """Compare consecutive numeric steps within a condition, never bridging gaps.

    Snapshot file names first so results written later by a collecting run are
    excluded from this analysis. Continuation JSON is saved atomically.
    """
    paths = sorted(sample_dir.glob("continuations/*/step_*.json"))
    records = {}
    for path in paths:
        record = load_json(path)
        condition, step = path.parent.name, int(path.stem.removeprefix("step_"))
        if record["condition"] != condition or record["step"] != step:
            raise ValueError(f"Continuation identity does not match its path: {path}")
        if record.get("status") not in (None, "complete"):
            continue
        records[condition, step] = record
    start, end = metadata["latent_step_start"], metadata["latent_steps"]
    conditions = sorted({condition for condition, _ in records})
    texts = {key: readout_text(record, tokenizer) for key, record in records.items()}
    excluded = {
        key
        for key, text in texts.items()
        if key[0] == "b"
        and re.fullmatch(r"(?P<math>\$?)\\boxed\{[^{}]+\}(?P=math)", text)
    }
    tokens = {
        key: tokenizer.encode(text, add_special_tokens=False)
        for key, text in texts.items()
        if key not in excluded
    }
    rows = []
    for (condition, step), record in sorted(records.items()):
        previous = records.get((condition, step - 1))
        if previous is None or not start < step <= end:
            continue
        if (condition, step) in excluded or (condition, step - 1) in excluded:
            continue
        left, right = tokens[condition, step - 1], tokens[condition, step]
        distance = token_levenshtein(left, right)
        rows.append(
            {
                "condition": condition,
                "previous_step": step - 1,
                "step": step,
                "edit_distance": distance,
                "normalized_distance": distance / max(len(left), len(right), 1),
                "previous_token_length": len(left),
                "token_length": len(right),
                "previous_correct": previous["correct"],
                "correct": record["correct"],
                "previous_budget_exhausted": previous["budget_exhausted"],
                "budget_exhausted": record["budget_exhausted"],
                "previous_prediction": previous["prediction"],
                "prediction": record["prediction"],
            }
        )
    return {
        "sample": sample_dir.name,
        "normalization": "collapse Unicode whitespace to one ASCII space; strip edges",
        "token_sequence": "decode content IDs without cleanup; encode without added special tokens",
        "terminal_eos": "excluded using content_length",
        "symbols_and_thinking_tags": "preserved",
        "distance": "unit-cost Levenshtein / max(token lengths); empty-empty = 0",
        "step_range": [start, end],
        "exclusion_policy": "B: normalized content consists solely of a boxed answer, optionally enclosed in $...$; exclude incident pairs without bridging",
        "excluded_steps": {
            c: sorted(k for condition, k in excluded if condition == c) for c in "abcd"
        },
        "available_steps": {
            c: sorted(k for condition, k in records if condition == c)
            for c in conditions
        },
        "missing_steps": {
            c: [k for k in range(start, end + 1) if (c, k) not in records]
            for c in "abcd"
        },
        "rows": rows,
    }


def analyze_readouts(directory, metadata, tokenizer, output):
    results = []
    for sample_dir in sorted((directory / "samples").iterdir()):
        if not sample_dir.is_dir():
            continue
        result = adjacent_readout_metrics(sample_dir, metadata, tokenizer)
        save_json_atomic(output / sample_dir.name / "token_distances.json", result)
        results.append(result)
    save_json_atomic(
        output / "source.json",
        {"directory": str(directory.resolve()), "metadata": metadata},
    )
    return results
