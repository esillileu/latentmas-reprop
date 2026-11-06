"""Token lengths and explicit thinking-boundary features for saved readouts."""


def readout_text(record, tokenizer):
    """Decode content without terminal EOS, retaining tags and symbols."""
    ids, length = record["generated_token_ids"], record["content_length"]
    if not 0 <= length <= len(ids):
        raise ValueError("Invalid continuation content_length")
    text = tokenizer.decode(
        ids[:length], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return " ".join(text.split())


def readout_features(record, markers):
    ids = record["generated_token_ids"]
    length = record["content_length"]
    if (
        type(record["correct"]) is not bool
        or not 0 <= length <= len(ids)
        or record["generated_length"] != len(ids)
    ):
        raise ValueError("Invalid correctness or token lengths")
    if not isinstance(ids, list) or any(
        type(token) is not int or token < 0 for token in ids
    ):
        raise ValueError("Invalid generated token IDs")
    content = ids[:length]
    opening, closing = markers
    # An opening prefill starts reasoning; a closing prefill ends it before output.
    reasoning = opening in record["inserted_prefix_ids"]
    if closing in record["inserted_readout_ids"]:
        reasoning = False
    reasoning_length = 0
    for token in content:
        if token == opening:
            reasoning = True
        elif token == closing:
            reasoning = False
        elif reasoning:
            reasoning_length += 1
    prefill = record["inserted_prefix_ids"] + record["inserted_readout_ids"]
    return {
        "evaluator_correct": record["correct"],
        "semantic_correct": None,
        "review_status": "unreviewed",
        "gold": record["gold"],
        "prediction": record["prediction"],
        "generated_length": len(ids),
        "content_length": length,
        "finish_reason": record["finish_reason"],
        "budget_exhausted": record["budget_exhausted"],
        "generated_think_open_count": content.count(opening),
        "generated_think_close_count": content.count(closing),
        "prefill_think_open_count": prefill.count(opening),
        "prefill_think_close_count": prefill.count(closing),
        "reasoning_token_count": reasoning_length,
        "reasoning_observation": "generated" if reasoning_length else "none",
    }
