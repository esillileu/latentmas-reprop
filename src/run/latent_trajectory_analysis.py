"""Analyze saved latent trajectories and whitespace-normalized readout tokens."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from latentmas_reprop.application.latent_trajectory_analysis import (
    analyze_readouts,
    analyze_trajectories,
)
from latentmas_reprop.infrastructure.models.trajectory_tokenizer import (
    load_trajectory_tokenizer,
)
from latentmas_reprop.infrastructure.storage import load_json
from latentmas_reprop.infrastructure.tracking.analysis_tracking import (
    log_analysis_results_to_mlflow,
)
from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker

from .latent_trajectory_report import write_report


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--continuations-only",
        action="store_true",
        help="Skip hidden/logit tensor analysis",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    metadata = load_json(directory / "metadata.json")
    tokenizer = load_trajectory_tokenizer(metadata)
    output = args.output_dir or (
        Path("artifacts/latent_trajectory")
        / metadata["task"]
        / directory.parent.name
        / directory.name
    )
    results = analyze_readouts(directory, metadata, tokenizer, output)
    write_report(results, output, directory, metadata)
    if not args.continuations_only:
        analyze_trajectories(directory)
    log_analysis_results_to_mlflow(
        MLflowTracker(),
        f"latent_trajectory/{metadata['task']}/{directory.parent.name}/{directory.name}",
        output,
        params={
            "source_directory": str(directory),
            "comparison": "adjacent",
            "preprocessing": "whitespace_only",
        },
        metrics={"pairs": sum(len(result["rows"]) for result in results)},
    )
    print(f"Analysis: {output.resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
