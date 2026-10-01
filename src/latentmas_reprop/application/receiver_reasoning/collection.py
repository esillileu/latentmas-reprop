"""Independent run reports with a model-labelled report index."""

import json
import re

from ..common.reporter import write_json
from .analysis import analyze_records
from .mlflow import load_run
from .report import export_analysis
from .tables import table


def analyze_runs(client, paths, run_ids, output, *, bootstrap_count=5000, seed=0):
    entries = []
    for run_id in run_ids:
        records, config, metadata = load_run(client, paths, run_id=run_id)
        metrics, matrix, distribution, bootstrap = analyze_records(
            records, config, metadata, bootstrap_count=bootstrap_count, seed=seed
        )
        model = config["model_name"]
        model_dir = re.sub(r"[^A-Za-z0-9_.-]", "_", model)
        directory = output / model_dir / run_id
        export_analysis(directory, metrics, matrix, distribution, bootstrap, records)
        entries.append(
            {
                "model": model,
                "run_id": run_id,
                "report": (directory / "summary.md").relative_to(output).as_posix(),
                "sample_counts": {
                    key: cell["sample_count"] for key, cell in metrics["cells"].items()
                },
            }
        )
        print(f"Analyzed {model}, MLflow run {run_id}: {directory / 'summary.md'}")
    write_json(output / "reports.json", entries)
    # The collection index is rendered from saved machine-readable metadata.
    saved = json.loads((output / "reports.json").read_text(encoding="utf-8"))
    text = "# Latent Handoff Pilot Analysis\n\nEach report analyzes one model/run independently. Results are not pooled across models or runs.\n\n"
    text += (
        table(
            ["model", "run", "cell sample counts", "report"],
            [
                [
                    row["model"],
                    row["run_id"],
                    row["sample_counts"],
                    f"[summary.md]({row['report']})",
                ]
                for row in saved
            ],
        )
        + "\n"
    )
    # Table values are escaped by display; links contain only sanitized paths.
    (output / "summary.md").write_text(text, encoding="utf-8")
    return entries
