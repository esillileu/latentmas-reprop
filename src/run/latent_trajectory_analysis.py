"""Report available latent trajectory results, including unfinished collection."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from latentmas_reprop.application.latent_alignment_analysis import compare_alignment
from latentmas_reprop.application.latent_evaluation_audit import (
    audit_evaluations,
    medqa_gold_audit,
)
from latentmas_reprop.application.latent_readout_analysis import (
    compare_conditions,
    load_readouts,
    summarize_readouts,
)
from latentmas_reprop.application.latent_readout_review import (
    apply_annotations,
    review_cases,
)
from latentmas_reprop.application.latent_trajectory_analysis import analyze_trajectories
from latentmas_reprop.infrastructure.models.trajectory_tokenizer import (
    load_trajectory_tokenizer,
)
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
from latentmas_reprop.infrastructure.storage import load_json
from latentmas_reprop.infrastructure.tracking.analysis_tracking import (
    log_analysis_results_to_mlflow,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker

from .latent_review_report import write_review_report
from .latent_state_report import write_state_report
from .latent_trajectory_report import write_csv, write_report


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--continuations-only", action="store_true")
    parser.add_argument("--skip-token-distance", action="store_true")
    parser.add_argument(
        "--track", action="store_true", help="Also log report to MLflow"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--compare-directory", type=Path, help="Compare an opposite-alignment source"
    )
    parser.add_argument(
        "--annotations", type=Path, help="Separate human annotation JSON array"
    )
    parser.add_argument(
        "--medqa-data", type=Path, help="Local MedQA source JSON for label audit"
    )
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    metadata = load_json(directory / "metadata.json")
    output = args.output_dir or (
        Path("artifacts/latent_trajectory")
        / metadata["task"]
        / directory.parent.name
        / directory.name
    )
    output.mkdir(parents=True, exist_ok=True)
    notes, tokenizer = [], None
    if not args.skip_token_distance:
        try:
            tokenizer = load_trajectory_tokenizer(metadata)
        except (OSError, ValueError) as error:
            notes.append(f"Token distance unavailable: {error}")
    else:
        notes.append("Token distance not requested.")
    rows, records = load_readouts(directory, metadata, tokenizer)
    annotation_status = apply_annotations(rows, args.annotations)
    evaluation_audit = audit_evaluations(rows, records, metadata["task"])
    gold_audit = []
    if metadata["task"] == "medqa":
        gold_audit = medqa_gold_audit(
            sorted((directory / "samples").glob("*/sample.json")),
            args.medqa_data or get_path_resolver().medqa_json_path,
        )
    trajectories, accuracies = summarize_readouts(rows)
    comparisons = compare_conditions(rows)
    cases = review_cases(rows, records)
    if args.compare_directory:
        other = args.compare_directory.resolve()
        other_metadata = load_json(other / "metadata.json")
        other_rows, _ = load_readouts(other, other_metadata)
        alignment = compare_alignment(rows, other_rows, metadata, other_metadata)
        write_csv(output / "alignment_comparisons.csv", alignment)
        notes.append(
            f"Alignment comparison: {other}; see alignment_comparisons.csv. Unpaired cells remain pending."
        )

    latent, statuses = [], []
    if not args.continuations_only:
        statuses, latent = analyze_trajectories(directory, output, tokenizer)
        unavailable = sum(s["status"] != "complete" for s in statuses)
        notes.append(
            f"Tensor analysis unavailable for {unavailable}/{len(statuses)} trajectories; see tensor_status.json."
        )
    else:
        notes.append("Tensor analysis not requested.")
    write_report(
        rows,
        trajectories,
        accuracies,
        comparisons,
        cases,
        output,
        directory,
        metadata,
        notes,
    )
    write_review_report(
        rows,
        cases,
        evaluation_audit,
        gold_audit,
        annotation_status,
        output,
        args.annotations,
    )
    if not args.continuations_only:
        write_state_report(rows, latent, statuses, output)
    if args.track:
        log_analysis_results_to_mlflow(
            MLflowTracker(),
            f"latent_trajectory/{metadata['task']}/{directory.parent.name}/{directory.name}",
            output,
            params={"source_directory": str(directory), "comparison": "paired_by_step"},
            metrics={
                "completed_readouts": sum(r["status"] == "complete" for r in rows)
            },
        )
    print(f"Analysis: {output.resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
