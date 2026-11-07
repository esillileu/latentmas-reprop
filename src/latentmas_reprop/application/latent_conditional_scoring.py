"""Teacher-forced answer candidates in an explicit, recorded output context."""

import torch
import torch.nn.functional as F


def answer_candidates(task, sample, records):
    """Never score MBPP test assertions as though they were an answer program."""
    if task in ("medqa", "arc_challenge"):
        return "\n\nFinal answer: \\boxed{", [
            {
                "text": letter + "}",
                "correct": letter.lower() == str(sample["gold"]).lower(),
                "origin": "choice_label",
            }
            for letter in "ABCD"
        ]
    candidates = {}
    for record in records:
        prediction = record.get("prediction")
        if prediction is not None and str(prediction).strip():
            text = str(prediction).strip()
            if task == "mbppplus":
                # Code is already extracted by the canonical evaluator.
                text += "\n```"
            else:
                text += "}"
            candidates.setdefault(text, set()).add(bool(record["correct"]))
    if task == "gsm8k":
        candidates[str(sample["gold"]).strip() + "}"] = {True}
    values = [
        {
            "text": text,
            "correct": next(iter(labels)),
            "origin": "saved_evaluator_prediction",
        }
        for text, labels in candidates.items()
        if len(labels) == 1
    ]
    # Two representative alternatives per class keep scoring code programs bounded.
    selected = []
    for label in (True, False):
        selected.extend(
            sorted(
                (v for v in values if v["correct"] == label),
                key=lambda v: (len(v["text"]), v["text"]),
            )[:2]
        )
    prefix = "\n\n```python\n" if task == "mbppplus" else "\n\nFinal answer: \\boxed{"
    return prefix, selected


@torch.inference_mode()
def score_candidate(
    model, cache, start_hidden, prefix_ids, candidate_ids, attention_mask=None
):
    """Return first-token and entire-string log P, excluding the forced prefix.

    The caller supplies a fresh working cache for each candidate. No EOS is
    appended; these are string probabilities, not a normalized answer posterior.
    """
    if not candidate_ids:
        raise ValueError("Candidate must contain at least one token")
    device = start_hidden.device
    inner = model.model
    h = start_hidden
    mask = attention_mask
    for offset in range(0, len(prefix_ids), 64):
        part = torch.tensor([prefix_ids[offset : offset + 64]], device=device)
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(part)], dim=-1)
        out = inner(
            input_ids=part,
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
        cache, h = out.past_key_values, out.last_hidden_state[0, -1]
    initial = F.linear(h, model.get_output_embeddings().weight).float().log_softmax(-1)
    first = initial[candidate_ids[0]].item()
    total = first
    for offset in range(0, len(candidate_ids) - 1, 32):
        inputs = candidate_ids[offset : min(offset + 32, len(candidate_ids) - 1)]
        targets = candidate_ids[offset + 1 : offset + 1 + len(inputs)]
        part = torch.tensor([inputs], device=device)
        if mask is not None:
            mask = torch.cat([mask, torch.ones_like(part)], dim=-1)
        out = inner(
            input_ids=part,
            past_key_values=cache,
            attention_mask=mask,
            use_cache=True,
            return_dict=True,
        )
        cache = out.past_key_values
        logs = (
            F.linear(out.last_hidden_state[0], model.get_output_embeddings().weight)
            .float()
            .log_softmax(-1)
        )
        total += (
            logs.gather(-1, torch.tensor(targets, device=device)[:, None]).sum().item()
        )
    return {
        "first_token_log_probability": first,
        "string_log_probability": total,
        "mean_token_log_probability": total / len(candidate_ids),
        "candidate_tokens": len(candidate_ids),
    }
