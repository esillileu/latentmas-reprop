"""Analyze saved KV structure without model inference or cache mutation."""

import argparse
from pathlib import Path

from latentmas_reprop.application.latent_kv_analysis import analyze_kv
from latentmas_reprop.infrastructure.storage import load_json, save_json_atomic

from .latent_kv_plots import plot_kv
from .latent_trajectory_report import write_csv


def write_kv_report(statuses, summaries, positions, output):
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "kv_layers.csv", summaries)
    write_csv(output / "kv_positions.csv", positions)
    (output / "kv_steps.csv").unlink(missing_ok=True)
    save_json_atomic(output / "kv_status.json", statuses)
    lines = [
        "# Saved KV structure",
        "",
        "Each A/C or B/D shared trajectory is read once. No full KV clone, new forward, model weights or attention queries are used.",
        "Vectors concatenate batch/heads at each cache position. K values retain the model's positional/RoPE encoding; they are not de-rotated.",
        "All prompt and latent cache positions are shown. The red line marks their boundary. Step 0 is the last prompt position; k>=1 is latent.",
        "Both regions use uncentered SVD of finite positions. Matched rank compares the last W prompt positions with the first W latent positions, W=min(prompt_length, latent_steps). Full rank also depends on region length.",
        "Prompt similarity searches valid nonzero prompt vectors. Prompt queries exclude themselves; latent queries search all prompt positions. Nearest indices are zero-based cache positions, not attention weights.",
        "Heatmap color ranges are shared between regions for each K/V metric. Equal-length matched ranks reuse full-region results when the windows coincide.",
        "Missing/invalid values are gray. Neither cache similarity nor low rank establishes reasoning, usefulness or information accumulation.",
        "",
        "| Sample | Alignment | Trajectory | Cache status |",
        "|---|---|---|---|",
    ]
    images = []
    for status in statuses:
        sample, trajectory = status["sample_id"], status["trajectory_type"]
        lines.append(
            f"| {sample} | {'ON' if status['alignment'] else 'OFF'} | {trajectory} | {status['status']} |"
        )
        if status.get("error"):
            lines.append(f"\nError: {status['error']}\n")
        selected = [
            r
            for r in positions
            if r["sample_id"] == sample and r["trajectory_type"] == trajectory
        ]
        layers = [
            r
            for r in summaries
            if r["sample_id"] == sample and r["trajectory_type"] == trajectory
        ]
        images.extend(plot_kv(selected, layers, sample, trajectory, output))
    for image in images:
        lines.extend(["", f"![KV structure]({image})", ""])
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--layers", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    metadata = load_json(directory / "metadata.json")
    output = (
        args.output_dir
        or Path("artifacts/latent_trajectory")
        / metadata["task"]
        / directory.parent.name
        / directory.name
        / "kv_analysis"
    )
    write_kv_report(*analyze_kv(directory, args.samples, args.layers), output)
    print(f"KV analysis: {output.resolve() / 'summary.md'}")


if __name__ == "__main__":
    main()
