"""Merge separately maintained human judgments without modifying collected data."""

from ..infrastructure.storage import load_json

REVIEW_FIELDS = (
    "semantic_correct",
    "reasoning_validity",
    "reason_label",
    "reasoning_progress",
    "restart",
    "review_notes",
)


def apply_annotations(rows, path):
    """Reject ambiguous identities and invalid values per entry, not per report."""
    if path is None:
        return []
    try:
        entries = load_json(path)
        if not isinstance(entries, list):
            raise ValueError("Annotations must be a JSON array")
    except (ValueError, OSError) as error:
        return [{"status": "invalid", "reason": str(error)}]
    lookup = {
        (r["source_run_id"], r["sample_id"], r["condition"], r["step"]): r for r in rows
    }
    keyed, results = {}, []
    for index, entry in enumerate(entries):
        try:
            key = tuple(
                entry[field]
                for field in ("source_run_id", "sample_id", "condition", "step")
            )
            if (
                not isinstance(key[0], str)
                or key[2] not in "abcd"
                or len(key[2]) != 1
                or type(key[1]) is not int
                or type(key[3]) is not int
                or key[1] < 0
                or key[3] < 0
            ):
                raise ValueError("Invalid annotation identity")
            hash(key)
            keyed.setdefault(key, []).append((index, entry))
        except (KeyError, TypeError, ValueError) as error:
            results.append({"entry": index, "status": "invalid", "reason": str(error)})
    for key, values in keyed.items():
        row = lookup.get(key)
        for index, entry in values:
            result = {
                "entry": index,
                "source_run_id": key[0],
                "sample_id": key[1],
                "condition": key[2],
                "step": key[3],
                "status": "invalid",
                "reason": None,
            }
            fields = {name: entry.get(name) for name in REVIEW_FIELDS}
            if len(values) > 1:
                result["reason"] = (
                    "Duplicate annotation identity; neither entry applied"
                )
            elif row is None or row["status"] != "complete":
                result.update(
                    status="unavailable", reason="No matching complete readout"
                )
            elif "gold" in entry and entry["gold"] != row["gold"]:
                result["reason"] = "Annotation gold differs from saved gold"
            elif any(
                fields[name] is not None and type(fields[name]) is not bool
                for name in ("semantic_correct", "restart")
            ):
                result["reason"] = (
                    "semantic_correct and restart must be boolean or null"
                )
            elif fields["reasoning_validity"] not in (
                None,
                "valid",
                "invalid",
                "no_reasoning",
                "undetermined",
            ):
                result["reason"] = "Invalid reasoning_validity"
            elif any(
                fields[name] is not None and not isinstance(fields[name], str)
                for name in ("reason_label", "reasoning_progress", "review_notes")
            ):
                result["reason"] = "Review labels/notes must be strings or null"
            else:
                reviewed = any(
                    value is not None and value != "" for value in fields.values()
                )
                row.update(
                    fields, review_status="reviewed" if reviewed else "unreviewed"
                )
                result["status"] = "applied" if reviewed else "unreviewed"
            results.append(result)
    return results


def review_template(rows, cases):
    """Generate both endpoints of transitions as editable annotation candidates."""
    selected = {
        (case["sample_id"], case["condition"], step)
        for case in cases
        for step in (case["step"] - 1, case["step"])
    }
    return [
        {
            **{
                key: row[key]
                for key in (
                    "source_run_id",
                    "sample_id",
                    "condition",
                    "step",
                    "gold",
                    "question",
                    "output_path",
                    "evaluator_correct",
                    "prediction",
                )
            },
            **dict.fromkeys(REVIEW_FIELDS),
        }
        for row in rows
        if row["status"] == "complete"
        and (row["sample_id"], row["condition"], row["step"]) in selected
    ]


def review_cases(rows, records):
    """Select adjacent correctness changes with both endpoints' human judgments."""
    lookup = {(r["sample_id"], r["condition"], r["step"]): r for r in rows}
    cases = []
    for row in rows:
        if row["status"] != "complete" or row["transition"] not in ("0→1", "1→0"):
            continue
        key = row["sample_id"], row["condition"], row["step"]
        previous = (*key[:2], key[2] - 1)
        before = lookup[previous]
        cases.append(
            {
                "source_run_id": row["source_run_id"],
                "sample_id": key[0],
                "condition": key[1],
                "step": key[2],
                "transition": row["transition"],
                "gold": row["gold"],
                "previous_output": records[previous]["generated_text"],
                "output": records[key]["generated_text"],
                "previous_review": {
                    field: before.get(field) for field in REVIEW_FIELDS
                },
                "review": {field: row.get(field) for field in REVIEW_FIELDS},
                "review_status": row["review_status"],
                "previous_review_status": before["review_status"],
            }
        )
    return cases
