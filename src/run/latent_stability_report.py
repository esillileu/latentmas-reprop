"""Descriptive output diagnostics; no language classifier or semantic judge."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .latent_trajectory_report import write_csv

STABILITY_FIELDS = (
    "source_run_id",
    "sample_id",
    "condition",
    "step",
    "status",
    "evaluator_correct",
    "repeated_4gram_fraction",
    "longest_identical_token_run",
    "repeated_line_fraction",
    "script_counts",
    "script_count",
    "script_switch_count",
    "non_latin_letter_fraction",
    "generated_chat_open_count",
    "odd_code_fence_count",
    "output_path",
)


def write_stability_report(rows, output):
    write_csv(
        output / "output_stability.csv",
        [{key: row.get(key) for key in STABILITY_FIELDS} for row in rows],
    )
    available = [r for r in rows if r["status"] == "complete"]
    lines = [
        "# Output repetition and script diagnostics",
        "",
        "Generated content IDs exclude terminal EOS. Repeated 4-grams count occurrences after the first, divided by all 4-gram windows.",
        "Line repetition uses exact whitespace-normalized nonempty lines, retaining case and punctuation.",
        "Unicode letter scripts are proxies, not language identification. Script switches ignore nonletters and exclude structural chat/thinking tags.",
        "Mixing scripts, repeating valid equations, or an odd fence count does not establish incoherence or incorrect reasoning.",
        "No LLM judge, semantic classifier or arbitrary failure threshold is used. Missing observations remain gaps.",
        "",
        f"Completed readouts: {len(available)}/{len(rows)}.",
        f"Readouts with multiple letter scripts: {sum(r['script_count'] > 1 for r in available)}.",
        f"Readouts generating chat-opening markers: {sum(r['generated_chat_open_count'] > 0 for r in available)}.",
        f"Readouts with an odd number of line-leading code fences: {sum(r['odd_code_fence_count'] for r in available)}.",
        "",
    ]
    for sample in sorted({r["sample_id"] for r in rows}):
        selected = [r for r in rows if r["sample_id"] == sample]
        steps = sorted({r["step"] for r in selected})
        lookup = {(r["condition"], r["step"]): r for r in selected}
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for axis, field, label in zip(
            axes,
            (
                "repeated_4gram_fraction",
                "longest_identical_token_run",
                "script_switch_count",
            ),
            (
                "Repeated 4-gram fraction",
                "Longest identical-token run",
                "Unicode script switches",
            ),
            strict=True,
        ):
            for condition in "abcd":
                values = [lookup[condition, step].get(field) for step in steps]
                axis.plot(
                    steps,
                    [np.nan if v is None else v for v in values],
                    label=condition.upper(),
                    linewidth=1,
                )
            axis.set(ylabel=label, facecolor="#eeeeee")
            axis.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("Latent step")
        fig.suptitle(f"Sample {sample}: observable output diagnostics")
        fig.tight_layout()
        image = f"output_stability_{sample:06d}.png"
        fig.savefig(output / image, dpi=150)
        plt.close(fig)
        lines.extend([f"![Sample {sample}]({image})", ""])
    (output / "output_stability_summary.md").write_text("\n".join(lines) + "\n")
    with (output / "summary.md").open("a") as stream:
        stream.write(
            "\n\n[Output repetition and script diagnostics](output_stability_summary.md)\n"
        )
