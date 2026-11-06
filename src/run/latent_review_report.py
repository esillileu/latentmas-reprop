"""Write audit evidence and human review templates alongside the main report."""

from collections import Counter

from latentmas_reprop.application.latent_readout_review import review_template
from latentmas_reprop.infrastructure.storage import save_json_atomic

from .latent_trajectory_report import write_csv


def write_review_report(
    rows, cases, evaluation_audit, gold_audit, annotations, output, annotation_path
):
    write_csv(output / "evaluation_audit.csv", evaluation_audit)
    save_json_atomic(output / "gold_audit.json", gold_audit)
    write_csv(output / "annotation_status.csv", annotations)
    template = output / "review_template.json"
    if annotation_path is None or annotation_path.resolve() != template.resolve():
        save_json_atomic(template, review_template(rows, cases))
    counts = Counter(row["status"] for row in evaluation_audit)
    labels = Counter(row["status"] for row in gold_audit)
    applied = Counter(row["status"] for row in annotations)
    lines = [
        "",
        "## Evaluator and source-label audit",
        "",
        "This checks recorded evaluation and source option mapping, not medical reasoning.",
        "",
        f"Evaluator replay: verified={counts['verified']}, mismatch={counts['mismatch']}, unavailable={counts['unavailable']}.",
        f"MedQA source mapping: verified={labels['verified']}, invalid={labels['invalid']}, unavailable={labels['unavailable']}.",
        "",
    ]
    if gold_audit:
        lines.extend(
            ["| Sample | Gold | Expected | Status | Reason |", "|---|---|---|---|---|"]
        )
        for row in gold_audit:
            lines.append(
                f"| {row.get('sample_id', '—')} | {row.get('stored_gold', '—')} | {row.get('expected_gold', '—')} | {row['status']} | {row.get('reason') or ''} |"
            )
        lines.append("")
    medqa = [r for r in evaluation_audit if "extraction_failed" in r]
    if medqa:
        lines.extend(
            [
                f"MedQA evaluator did not extract A-D: {sum(r['extraction_failed'] for r in medqa)}/{len(medqa)}.",
                f"Explicit final-choice evidence disagrees with stored prediction: {sum(r['choice_disagrees'] for r in medqa)}.",
                "Explicit-choice detection is conservative diagnostic evidence, never a replacement score.",
                "",
            ]
        )
    lines.extend(
        [
            "## Human review",
            "",
            f"Annotations: applied={applied['applied']}, unreviewed={applied['unreviewed']}, invalid={applied['invalid']}, unavailable={applied['unavailable']}.",
            "Copy review_template.json to a separate annotations.json, edit the judgments and pass --annotations annotations.json.",
            "Identity fields must match the source run, item, condition and step. Duplicate entries are not applied.",
            "semantic_correct/restart: boolean or null. reasoning_validity: valid, invalid, no_reasoning, undetermined, or null.",
            "reason_label, reasoning_progress and review_notes are optional human text. Blank templates remain unreviewed.",
            "Generated templates/candidates may be refreshed; supplied annotation files are preserved.",
            "",
        ]
    )
    with (output / "summary.md").open("a") as stream:
        stream.write("\n".join(lines))
