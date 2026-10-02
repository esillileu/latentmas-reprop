"""Serialize analysis data and render Markdown only from exported artifacts."""

import json
from pathlib import Path

from ..common.reporter import write_json, write_jsonl
from .execution_report import execution_table
from .tables import display, read_csv, table, write_csv


def sample_table(rows):
    keys = (
        [key.removesuffix("_correct") for key in rows[0] if key.endswith("_correct")]
        if rows
        else ["low_answer_only", "low_free", "high_answer_only", "high_free"]
    )
    values = []
    for row in rows:
        cells = []
        for key in keys:
            correct = row[f"{key}_correct"]
            cells.append(
                "N/A"
                if correct is None
                else f"{row[f'{key}_prediction']} / {correct} / {row[f'{key}_receiver_generated_tokens']}"
            )
        values.append([row["sample_index"], row["gold"], *cells])
    return table(["sample", "gold", *keys], values)


def render_markdown(output: Path):
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    bootstrap = json.loads(
        (output / "bootstrap_statistics.json").read_text(encoding="utf-8")
    )
    matrix = read_csv(output / "sample_matrix.csv")
    distribution = read_csv(output / "prediction_distribution.csv")
    sample_labels = {row["sample_id"]: row["sample_index"] for row in matrix}
    cells, derived = metrics["cells"], metrics["derived"]
    sections = ["# Latent Handoff Pilot Analysis", "## Run Configuration"]
    run = metrics["run"]
    sections += [
        table(
            ["field", "value"],
            [
                [k, run[k]]
                for k in (
                    "run_id",
                    "experiment_id",
                    "status",
                    "start_time",
                    "end_time",
                    "artifact_uri",
                )
            ],
        ),
        table(["configuration", "value"], list(metrics["config"].items())),
        "MLflow recorded metrics",
        table(["metric", "value"], list(run["metrics"].items())),
        table(
            ["bootstrap setting", "value"],
            [
                [k, bootstrap[k]]
                for k in ("method", "confidence_level", "bootstrap_count", "seed")
            ],
        ),
        "## Data Integrity",
    ]
    integrity = metrics["integrity"]
    sections += [
        table(
            ["check", "value"],
            [
                ["all sample sets equal", integrity["all_sample_sets_equal"]],
                ["issues", integrity["issues"]],
            ],
        ),
        table(
            ["cell", "sample indices", "problems"],
            [
                [
                    key,
                    [sample_labels[sid] for sid in ids],
                    integrity["cell_problems"][key],
                ]
                for key, ids in integrity["cell_sample_ids"].items()
            ],
        ),
        "Paired metrics use complete equal sample sets; invalid comparisons are N/A.",
        "## Experiment Matrix",
        table(
            ["cell", "samples", "correct", "accuracy"],
            [
                [k, c["sample_count"], c["correct_count"], c["accuracy"]]
                for k, c in cells.items()
            ],
        ),
        "## Receiver Reasoning Comparison",
        "D(S) = Acc(S, free) - Acc(S, answer_only)",
    ]
    levels = [level for level in ("low", "high") if f"{level}_free" in cells]
    sections.append(
        table(
            ["metric", *levels],
            [
                [
                    "answer-only accuracy",
                    *[cells[f"{v}_answer_only"]["accuracy"] for v in levels],
                ],
                ["free accuracy", *[cells[f"{v}_free"]["accuracy"] for v in levels]],
                ["D(S)", *[derived[f"D_{v}"]["estimate"] for v in levels]],
                ["D(S) 95% CI", *[derived[f"D_{v}"]["ci95"] for v in levels]],
            ],
        )
    )
    for title, names in (
        ("Receiver paired transitions", ("D_low", "D_high")),
        ("Upstream Compute Comparison", ("Delta_answer_only", "Delta_free")),
    ):
        names = [name for name in names if name in derived]
        if not names:
            continue
        sections.append(
            "### " + title if title == "Receiver paired transitions" else "## " + title
        )
        sections.append(
            table(
                [
                    "metric",
                    "estimate",
                    "95% CI",
                    "paired n",
                    "both correct",
                    "free only correct"
                    if title == "Receiver paired transitions"
                    else "high only correct",
                    "answer_only only correct"
                    if title == "Receiver paired transitions"
                    else "low only correct",
                    "both wrong",
                    "positive cell",
                    "negative cell",
                    "unavailable reasons",
                ],
                [
                    [
                        name,
                        derived[name]["estimate"],
                        derived[name]["ci95"],
                        derived[name]["paired_sample_count"],
                        *(
                            (derived[name]["paired_transitions"] or {}).get(key)
                            for key in (
                                "both_correct",
                                "positive_only_correct",
                                "negative_only_correct",
                                "both_wrong",
                                "positive_cell",
                                "negative_cell",
                            )
                        ),
                        derived[name]["unavailable_reasons"],
                    ]
                    for name in names
                ],
            )
        )
    sections.append("## Combined Metrics")
    if "substitution_signal" in derived:
        signal = derived["substitution_signal"]
        sections += [
            "substitution_signal = D(low) - D(high)",
            f"substitution_signal = {display(signal['estimate'])}",
            f"95% CI = {display(signal['ci95'])}",
        ]
    else:
        sections.append("Single upstream level: compute contrasts are not estimated.")
    sections += [
        table(
            ["metric", "estimate", "95% CI", "paired n", "unavailable reasons"],
            [
                [
                    name,
                    v["estimate"],
                    v["ci95"],
                    v["paired_sample_count"],
                    v["unavailable_reasons"],
                ]
                for name, v in derived.items()
            ],
        ),
        "## Generation Diagnostics",
        table(
            [
                "cell",
                "mean tokens",
                "median",
                "min",
                "max",
                "parse failures",
                "empty outputs",
                "truncations",
            ],
            [
                [
                    k,
                    *(
                        c["generated_tokens"][name]
                        for name in ("mean", "median", "min", "max")
                    ),
                    c["parse_failure_count"],
                    c["empty_output_count"],
                    c["truncation_count"],
                ]
                for k, c in cells.items()
            ],
        ),
        "Truncation denotes reaching the configured generation limit; parse failure denotes missing or empty parsed prediction.",
        "## Execution Diagnostics",
        execution_table(cells),
        "## Prediction Distribution",
        table(
            ["cell", "unique predictions", "top prediction", "count", "ratio"],
            [
                [
                    k,
                    c["unique_prediction_count"],
                    *(
                        (c["top_prediction"] or {}).get(name)
                        for name in ("prediction", "count", "ratio")
                    ),
                ]
                for k, c in cells.items()
            ],
        ),
    ]
    for key in cells:
        sections += [
            "### " + key + " top 10",
            table(
                ["prediction", "count", "ratio"],
                [
                    [r["prediction"], r["count"], r["ratio"]]
                    for r in distribution
                    if r["cell"] == key
                ][:10],
            ),
        ]
    sections += [
        "## Sample-level Paired Results",
        "Samples use the original zero-based sample_index, unique within this run. Original sample IDs remain in sample_matrix.csv and JSON artifacts.\n\nCell format: prediction / correctness / generated_tokens",
        sample_table(matrix),
    ]
    for name, subset in metrics["subsets"].items():
        ids = {v["sample_id"] if isinstance(v, dict) else v for v in subset}
        sections += [
            "### " + name,
            sample_table([row for row in matrix if row["sample_id"] in ids]),
        ]
        if name == "repeated_predictions":
            sections += [
                metrics["repeated_prediction_definition"],
                table(
                    ["sample", "cells"],
                    [[sample_labels[r["sample_id"]], r["cells"]] for r in subset],
                ),
            ]
    sections += [
        "## Selected Raw Outputs",
        "Full outputs: raw_outputs.jsonl. Excerpts are limited to 1200 characters.",
    ]
    for example in metrics["selected_raw_outputs"]:
        sections += [
            table(
                ["cell", "sample", "correctness", "excerpt truncated"],
                [
                    [
                        example["cell"],
                        sample_labels[example["sample_id"]],
                        example.get("correct"),
                        example["excerpt_truncated"],
                    ]
                ],
            ),
            "<pre>" + display(example["raw_output"]).replace("<br>", "\n") + "</pre>",
        ]
    return "\n\n".join(sections) + "\n"


def export_analysis(output, metrics, matrix, distribution, bootstrap, records):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "metrics.json", metrics)
    write_json(output / "bootstrap_statistics.json", bootstrap)
    write_csv(output / "sample_matrix.csv", matrix)
    write_csv(output / "prediction_distribution.csv", distribution)
    write_jsonl(output / "raw_outputs.jsonl", records)
    (output / "summary.md").write_text(render_markdown(output), encoding="utf-8")
