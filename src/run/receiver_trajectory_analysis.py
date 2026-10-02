"""Analyze saved receiver trajectories and probe cells without model inference."""

import argparse
import csv
import json
from pathlib import Path

from latentmas_reprop.application.receiver_acquisition.trajectory import (
    summarize_trajectory,
)

from .presentation_plots import STYLE, plt
from .receiver_trajectory_plots import plot_trajectory


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_trajectory(input_dir, probe_path, output_dir):
    with probe_path.open(newline="") as stream:
        probes = list(csv.DictReader(stream))
    combined = []
    for size in ("0.6B", "4B", "8B", "14B"):
        model = f"Qwen/Qwen3-{size}"
        directory = input_dir / model.replace("/", "_")
        path = directory / "sample_results.jsonl"
        if not path.exists():
            continue
        source = json.loads((directory / "source.json").read_text())
        parity = json.loads((directory / "parity.json").read_text())
        if source["smoke"] or source["sample_count"] != 100 or not parity["passed"]:
            raise ValueError(f"Incomplete or unverified trajectory: {model}")
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        rows = summarize_trajectory(records, probes, model)
        combined.extend(rows)
    if not combined:
        raise ValueError("No collected receiver trajectories found")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trajectory.csv", combined)
    (output_dir / "trajectory.json").write_text(json.dumps(combined, indent=2) + "\n")
    with plt.rc_context(STYLE):
        plot_trajectory(output_dir / "trajectory.csv", output_dir / "plots")
    return combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("artifacts/receiver_trajectory")
    )
    parser.add_argument(
        "--probe-cells",
        type=Path,
        default=Path("artifacts/receiver_acquisition/sender_probe_cells.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/receiver_trajectory/analysis"),
    )
    args = parser.parse_args(argv)
    analyze_trajectory(args.input_dir, args.probe_cells, args.output_dir)


if __name__ == "__main__":
    main()
