def evaluate_predictions(preds: list[dict]) -> tuple[float, int]:
    """Calculate accuracy and correct sample count from predictions."""
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct
