"""Tables and partial-result plots for latent trajectory readouts."""

import csv
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from latentmas_reprop.infrastructure.storage import save_json_atomic


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows,
    trajectories,
    accuracies,
    comparisons,
    cases,
    output,
    directory,
    metadata,
    notes,
):
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("readouts", rows),
        ("trajectories", trajectories),
        ("step_accuracy", accuracies),
        ("condition_comparisons", comparisons),
    ):
        write_csv(output / f"{name}.csv", data)
    # Generated candidates are separate from any future human annotation file.
    save_json_atomic(output / "review_candidates.json", cases)
    save_json_atomic(
        output / "source.json",
        {"directory": str(directory), "metadata": metadata, "notes": notes},
    )
    counts = Counter(row["status"] for row in rows)
    lines = [
        "# Latent trajectory readouts",
        "",
        f"Source: `{directory}`",
        "",
        f"Model: {metadata['model_name']}; alignment: {metadata['latent_space_realign']}",
        "",
        f"Completed: {counts['complete']}/{len(rows)}; missing: {counts['missing']}; incomplete: {counts['incomplete']}; invalid: {counts['invalid']}.",
        "",
        "Unavailable observations are gray, never counted as incorrect. Curves do not bridge gaps.",
        "Step accuracy uses available items; see step_accuracy.csv for denominators and item IDs.",
        "Within-item correct-step fractions are descriptive, not independent-item accuracy.",
        f"Human-reviewed readouts: {sum(r['review_status'] == 'reviewed' for r in rows)}; all others remain unreviewed. No LLM judge is used.",
        "Condition comparisons use paired items at each step. CSVs include exact McNemar p-values and conservative 95% paired-difference intervals (Bonferroni Clopper-Pearson discordant cells). These are exploratory pointwise intervals for selected items, without multiplicity adjustment across steps/conditions; they do not establish population accuracy or significance. Repeated steps are never pooled as independent items.",
        "Reasoning length counts tokens inside explicit thinking boundaries; it is not a semantic reasoning measure.",
        "",
    ]
    lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "| Sample | Condition | Completed | Correct-step fraction | Gains | Losses | Pattern |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for item in trajectories:
        fraction = item["correct_step_fraction"]
        value = "—" if fraction is None else f"{fraction:.3f}"
        lines.append(
            f"| {item['sample_id']} | {item['condition']} | {item['completed']}/{item['expected']} | {value} | {item['wrong_to_correct']} | {item['correct_to_wrong']} | {item['pattern']} |"
        )
    colors = ListedColormap(["#bdbdbd", "#d95f59", "#54a86b"])
    for sample in sorted({row["sample_id"] for row in rows}):
        selected = [r for r in rows if r["sample_id"] == sample]
        steps = list(range(metadata["latent_step_start"], metadata["latent_steps"] + 1))
        lookup = {(r["condition"], r["step"]): r for r in selected}
        fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
        matrix = [
            [
                int(lookup[c, k]["evaluator_correct"])
                if lookup[c, k]["status"] == "complete"
                else -1
                for k in steps
            ]
            for c in "abcd"
        ]
        axes[0].imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap=colors,
            norm=BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3),
            extent=(steps[0] - 0.5, steps[-1] + 0.5, 3.5, -0.5),
        )
        axes[0].set_yticks(range(4), list("ABCD"))
        axes[0].legend(
            handles=[
                Patch(color=color, label=label)
                for color, label in zip(
                    colors.colors, ("Unavailable", "Incorrect", "Correct"), strict=True
                )
            ],
            loc="upper right",
        )
        for condition in "abcd":
            for axis, field in zip(
                axes[1:],
                ("content_length", "reasoning_token_count", "adjacent_token_distance"),
                strict=True,
            ):
                values = [
                    lookup[condition, k].get(field)
                    if lookup[condition, k]["status"] == "complete"
                    else None
                    for k in steps
                ]
                axis.plot(
                    steps,
                    [np.nan if v is None else v for v in values],
                    label=condition.upper(),
                )
                axis.set_facecolor("#eeeeee")
                axis.grid(alpha=0.2)
        for axis, label in zip(
            axes[1:],
            ("Content tokens", "Thinking-boundary tokens", "Adjacent token distance"),
            strict=True,
        ):
            axis.set_ylabel(label)
        axes[1].legend()
        axes[-1].set_xlabel("Latent step")
        fig.suptitle(f"{metadata['task']} | {metadata['model_name']} | sample {sample}")
        fig.tight_layout()
        name = f"sample_{sample:06d}.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        lines.extend(["", f"![Sample {sample}]({name})", ""])
    lines.extend(
        [
            "",
            "## Correctness transition review",
            "",
            f"Candidates: {len(cases)}. Full outputs: review_candidates.json.",
            "",
        ]
    )
    for case in cases:
        lines.extend(
            [
                f"### Sample {case['sample_id']} · {case['condition'].upper()} · {case['step'] - 1} → {case['step']} ({case['transition']})",
                "",
                f"Review before/after: {case.get('previous_review_status', 'unreviewed')} / {case['review_status']}",
                "",
                f"Human judgments before: {case['previous_review']}",
                f"Human judgments after: {case['review']}",
                "",
            ]
        )
        for label, field in (("Before", "previous_output"), ("After", "output")):
            # Indented blocks safely display arbitrary model-produced fences.
            lines.extend(
                [
                    f"{label} (first 2000 characters):",
                    "",
                    *["    " + line for line in case[field][:2000].splitlines()],
                    "",
                ]
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
