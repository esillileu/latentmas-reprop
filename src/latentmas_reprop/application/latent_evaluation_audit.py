"""Audit saved evaluator decisions and MedQA labels without medical judgments."""

import re

from ..infrastructure.evaluators.evaluator import StandardEvaluator, normalize_answer
from ..infrastructure.storage import load_json


def medqa_gold_audit(sample_paths, dataset_path):
    """Validate the loader's positional mapping against exact option text."""
    if not dataset_path.is_file():
        return [{"status": "unavailable", "reason": f"Missing dataset: {dataset_path}"}]
    try:
        dataset = load_json(dataset_path)
    except (ValueError, OSError) as error:
        return [{"status": "invalid", "reason": str(error)}]
    rows = []
    for path in sample_paths:
        sample = load_json(path)
        row = {
            "sample_id": sample["index"],
            "stored_gold": sample["gold"],
            "source_path": str(dataset_path.resolve()),
            "status": "invalid",
            "reason": None,
        }
        try:
            item = dataset[sample["index"]]
            options, answer = item["options"], str(item["answer"]).strip()
            # Remove only an explicit option prefix, not arbitrary answer text.
            option_texts = [
                re.sub(r"^[A-D][.)]\s*", "", option).strip() for option in options
            ]
            exact = [i for i, text in enumerate(option_texts) if text == answer]
            substring = [i for i, option in enumerate(options) if answer in option]
            row.update(
                raw_answer=answer,
                options=options,
                exact_matches=exact,
                substring_matches=substring,
            )
            if item["query"] != sample["question"]:
                row["reason"] = "Source question differs from saved sample"
            elif len(options) != 4 or len(exact) != 1:
                row["reason"] = "Gold does not identify exactly one of four options"
            else:
                expected = "abcd"[exact[0]]
                row["expected_gold"] = expected
                if substring != exact:
                    row["reason"] = (
                        "Loader substring mapping is ambiguous or differs from exact answer"
                    )
                elif normalize_answer(sample["gold"]) != expected:
                    row["reason"] = "Stored gold differs from source option"
                else:
                    row["status"] = "verified"
        except (IndexError, KeyError, TypeError) as error:
            row["reason"] = str(error)
        rows.append(row)
    return rows


def explicit_choice(text):
    """Extract only explicit boxed choices or an unambiguous final answer line.

    Diagnostic evidence only: never replaces the canonical evaluator or infers
    an answer from numbers, medical claims, or mentions of options in reasoning.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        match = re.fullmatch(r"\s*\(?([A-D])\)?\s*", boxes[-1], re.IGNORECASE)
        return match[1].lower() if match else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    match = re.fullmatch(
        r"(?:(?:the\s+)?(?:final\s+)?(?:answer|correct\s+(?:answer|option))\s*(?:is|:)\s*)?\(?([A-D])\)?[.]?",
        lines[-1],
        re.IGNORECASE,
    )
    return match[1].lower() if match else None


def audit_evaluations(rows, records, task):
    """Re-evaluate saved evaluation_text and keep diagnostic discrepancies visible."""
    evaluator, results = StandardEvaluator(), []
    for row in rows:
        if row["status"] != "complete":
            continue
        key = row["sample_id"], row["condition"], row["step"]
        record = records[key]
        result = {
            "source_run_id": row["source_run_id"],
            "sample_id": key[0],
            "condition": key[1],
            "step": key[2],
            "stored_prediction": record["prediction"],
            "stored_correct": record["correct"],
            "status": "unavailable",
        }
        if "evaluation_text" in record:
            text = record["evaluation_text"]
            prediction, correct, error = evaluator.evaluate(task, text, record["gold"])
            result.update(
                recomputed_prediction=prediction,
                recomputed_correct=correct,
                evaluator_error=error,
                status="verified"
                if prediction == record["prediction"] and correct == record["correct"]
                else "mismatch",
            )
            if task == "medqa":
                choice = explicit_choice(text)
                result.update(
                    explicit_choice=choice,
                    choice_disagrees=choice is not None
                    and choice != record["prediction"],
                    extraction_failed=prediction not in ("a", "b", "c", "d"),
                )
        results.append(result)
    return results
