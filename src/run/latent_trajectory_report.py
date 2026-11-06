"""Render adjacent readout distances as tables, curves and output excerpts."""

import csv
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from latentmas_reprop.infrastructure.storage import load_json


def write_report(results, output, directory, metadata):
    lines = [
        "# Adjacent token-level Levenshtein distances",
        "",
        f"Source: `{directory}`",
        "",
        f"Analyzed at: {datetime.now(UTC).isoformat()}",
        "",
        f"Model: {metadata['model_name']}; alignment: {metadata['latent_space_realign']}",
        "",
        "Whitespace only: collapse whitespace to one space and strip edges. "
        "Retokenize with the recorded tokenizer; preserve symbols and thinking tags; exclude terminal EOS. "
        "Distance = edit distance / maximum token length. Missing steps are not bridged.",
        "",
        "B readouts consisting solely of a boxed answer are excluded regardless of correctness. "
        "Pairs touching excluded steps are omitted; their neighbors are not connected.",
        "",
        "Peaks identify candidate output changes, including content changes; "
        "they do not alone establish a reasoning or formatting transition. "
        "Sampling is enabled in these runs. Inspect the paired outputs and truncation flags.",
        "",
    ]
    for result in results:
        sample = result["sample"]
        rows = result["rows"]
        lines.extend([f"## Sample {sample}", "", f"Compared pairs: {len(rows)}.", ""])
        lines.extend(
            [
                f"Excluded B readout steps: {result['excluded_steps']['b'] or 'none'}.",
                "",
            ]
        )
        if not rows:
            for name in ("token_distances.csv", "adjacent_distances.png"):
                (output / sample / name).unlink(missing_ok=True)
            lines.extend(["No adjacent completed readouts available.", ""])
            continue
        sample_output = output / sample
        with (sample_output / "token_distances.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True, sharey=True)
        start, end = result["step_range"]
        for condition, axis in zip("abcd", axes, strict=True):
            selected = [row for row in rows if row["condition"] == condition]
            values = {row["step"]: row["normalized_distance"] for row in selected}
            steps = list(range(start + 1, end + 1))
            axis.plot(
                steps,
                [values.get(k, float("nan")) for k in steps],
                marker=".",
                linewidth=1,
            )
            axis.set(ylabel=condition.upper(), ylim=(-0.02, 1.02))
            axis.grid(alpha=0.25)
            if not selected:
                lines.extend([f"### {condition.upper()}", "", "No adjacent pairs.", ""])
                continue
            ranked = sorted(
                selected, key=lambda row: (-row["normalized_distance"], row["step"])
            )
            peak = ranked[0]
            axis.annotate(
                f"{peak['previous_step']}→{peak['step']}",
                (peak["step"], peak["normalized_distance"]),
                fontsize=9,
            )
            mean = sum(row["normalized_distance"] for row in selected) / len(selected)
            lines.extend(
                [
                    f"### {condition.upper()}",
                    "",
                    f"Pairs: {len(selected)}/{end - start}; mean distance: {mean:.4f}; "
                    f"missing readout steps: {result['missing_steps'][condition] or 'none'}.",
                    f"Excluded readout steps: {result['excluded_steps'][condition] or 'none'}.",
                    "",
                    "| Step | Distance | Tokens before → after | Correct before → after | Budget exhausted before → after |",
                    "|---|---:|---|---|---|",
                ]
            )
            for row in ranked[:3]:
                lines.append(
                    f"| {row['previous_step']} → {row['step']} | {row['normalized_distance']:.4f} "
                    f"| {row['previous_token_length']} → {row['token_length']} "
                    f"| {row['previous_correct']} → {row['correct']} "
                    f"| {row['previous_budget_exhausted']} → {row['budget_exhausted']} |"
                )
            lines.extend(
                [
                    "",
                    "Largest-distance pair, original output excerpts (first 800 characters):",
                    "",
                ]
            )
            for step in (peak["previous_step"], peak["step"]):
                path = (
                    directory
                    / "samples"
                    / sample
                    / "continuations"
                    / condition
                    / f"step_{step:03d}.json"
                )
                record = load_json(path)
                lines.extend(
                    [
                        f"Step {step}: [{path.name}]({path})",
                        "",
                        "````text",
                        record["generated_text"][:800],
                        "````",
                        "",
                    ]
                )
        axes[-1].set_xlabel("Latent step k (comparison: k-1 → k)")
        fig.suptitle(
            f"{metadata['model_name']} | {metadata['task']} sample {sample}: "
            "adjacent normalized token edit distance"
        )
        fig.tight_layout()
        fig.savefig(sample_output / "adjacent_distances.png", dpi=200)
        plt.close(fig)
        lines.extend([f"![Sample {sample}]({sample}/adjacent_distances.png)", ""])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
