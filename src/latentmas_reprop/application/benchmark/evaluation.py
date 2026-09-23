"""Prediction aggregation for benchmark runs."""

OUTPUT_TOKEN_DEFINITION = (
    "Sum of newly generated text token ids across agents, excluding the prompt "
    "and pad ids. Latent rollout steps are counted separately and are not added. "
    "The released LatentMAS runner does not aggregate tokens. Table 1 Token is "
    "described as total output token usage, so these figures are not a verified "
    "match to that column."
)

WALL_TIME_DEFINITION = (
    "eval_time_sec times the evaluation loop after the model has loaded, which "
    "is where the original LatentMAS run.py starts total_time. "
    "model_load_time_sec is stored separately and is excluded. Table 1 Speed is "
    "end-to-end time in seconds per run on the authors' hardware."
)


def generated_token_count(pred: dict) -> int:
    """Text tokens produced for one sample. Latent steps are ignored."""
    if isinstance(pred.get("generated_tokens"), int):
        return int(pred["generated_tokens"])
    total = 0
    for agent in pred.get("agents") or []:
        count = agent.get("generated_tokens")
        if isinstance(count, int):
            total += count
    return total


def latent_step_count(pred: dict) -> int:
    """Latent rollout steps recorded for one sample."""
    if isinstance(pred.get("latent_steps_executed"), int):
        return int(pred["latent_steps_executed"])
    total = 0
    for agent in pred.get("agents") or []:
        steps = agent.get("latent_steps")
        if isinstance(steps, int):
            total += steps
    return total


def evaluate_predictions(preds: list[dict]) -> tuple[float, int]:
    """Calculate accuracy and correct sample count from predictions."""
    summary = summarize_predictions(preds)
    return summary["accuracy"], summary["correct"]


def summarize_predictions(preds: list[dict]) -> dict[str, float | int]:
    """Aggregate accuracy, text tokens, and latent steps."""
    total = len(preds)
    correct = sum(1 for pred in preds if pred.get("correct", False))
    output_tokens = sum(generated_token_count(pred) for pred in preds)
    latent_steps = sum(latent_step_count(pred) for pred in preds)
    return {
        "accuracy": (correct / total) if total else 0.0,
        "correct": correct,
        "total": total,
        "output_tokens_total": output_tokens,
        "output_tokens_mean": (output_tokens / total) if total else 0.0,
        "latent_steps_total": latent_steps,
        "latent_steps_mean": (latent_steps / total) if total else 0.0,
    }
