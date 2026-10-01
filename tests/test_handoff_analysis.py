import copy
import json

import numpy as np
import pytest
import yaml

from latentmas_reprop.application.receiver_reasoning.analysis import analyze_records
from latentmas_reprop.application.receiver_reasoning.mlflow import load_run, select_runs
from latentmas_reprop.application.receiver_reasoning.report import (
    export_analysis,
    render_markdown,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver


def test_sample_paired_bootstrap_and_transition_counts(pilot):
    m, matrix, distribution, bootstrap = analyze_records(
        *pilot, bootstrap_count=5000, seed=3
    )
    assert not m["integrity"]["issues"]
    assert len(matrix) == 4
    assert m["derived"]["D_low"]["estimate"] == 0.5
    assert m["derived"]["D_high"]["estimate"] == 0.5
    assert m["derived"]["Delta_answer_only"]["estimate"] == 0
    assert m["derived"]["Delta_free"]["estimate"] == 0
    assert m["derived"]["substitution_signal"]["estimate"] == 0
    transitions = m["derived"]["D_low"]["paired_transitions"]
    assert [
        transitions[k]
        for k in [
            "both_correct",
            "positive_only_correct",
            "negative_only_correct",
            "both_wrong",
        ]
    ] == [1, 2, 0, 1]
    samples = bootstrap["metrics"]
    np.testing.assert_allclose(
        samples["substitution_signal"]["replicates"],
        np.array(samples["D_low"]["replicates"])
        - np.array(samples["D_high"]["replicates"]),
    )
    np.testing.assert_allclose(
        m["derived"]["substitution_signal"]["ci95"],
        np.quantile(samples["substitution_signal"]["replicates"], [0.025, 0.975]),
    )
    assert m["subsets"]["answer_only_wrong_to_correct"] == ["0"]
    assert m["subsets"]["answer_only_correct_to_wrong"] == ["1"]
    assert sum(r["count"] for r in distribution if r["cell"] == "low_answer_only") == 4
    assert bootstrap == analyze_records(*pilot, bootstrap_count=5000, seed=3)[3]


def test_identical_paired_cells_have_zero_width_ci(pilot):
    rows, config, metadata = pilot
    for row in rows:
        row["correct"] = row["sample_id"] in {"0", "2"}
    m = analyze_records(rows, config, metadata, bootstrap_count=100)[0]
    assert all(metric["ci95"] == [0, 0] for metric in m["derived"].values())


def test_sample_mismatch_never_uses_intersection(pilot):
    rows, config, metadata = pilot
    rows.pop()
    m, matrix, _, bootstrap = analyze_records(rows, config, metadata)
    assert len(matrix) == 4
    assert m["derived"]["D_low"]["paired_sample_count"] == 4
    for key in ["D_high", "Delta_free", "substitution_signal"]:
        assert m["derived"][key]["estimate"] is None
        assert bootstrap["metrics"][key]["replicates"] == []
    assert m["subsets"]["high_disagreement"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "seed",
        "context",
        "gold",
        "params",
        "missing_config",
        "index_collision",
        "index_inconsistent",
    ],
)
def test_integrity_errors_make_affected_metrics_unavailable(pilot, mutation):
    rows, config, metadata = pilot
    if mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif mutation == "seed":
        rows[0]["config"]["seed"] = 100
    elif mutation == "context":
        rows[0]["context_id"] = "different"
    elif mutation == "gold":
        rows[0]["gold"] = "different"
    elif mutation == "index_collision":
        for row in rows:
            row["sample_index"] = 0
    elif mutation == "index_inconsistent":
        rows[0]["sample_index"] = 99
    elif mutation == "params":
        metadata["params"]["model_name"] = "different"
    else:
        del rows[0]["config"]["temperature"]
    m = analyze_records(rows, config, metadata)[0]
    assert m["integrity"]["issues"]
    assert m["derived"]["D_low"]["estimate"] is None
    assert m["derived"]["substitution_signal"]["estimate"] is None


def test_diagnostics_and_machine_readable_rendering(pilot, tmp_path):
    rows, config, metadata = pilot
    rows[0].update(
        prediction=None,
        raw_receiver_output="",
        receiver_generated_tokens=64,
        receiver_token_limit_reached=True,
    )
    results = analyze_records(rows, config, metadata, bootstrap_count=100)
    export_analysis(tmp_path, *results, rows)
    cell = results[0]["cells"]["low_answer_only"]
    assert (
        cell["parse_failure_count"],
        cell["empty_output_count"],
        cell["truncation_count"],
    ) == (1, 1, 1)
    assert cell["generated_tokens"] == {
        "mean": 19.75,
        "median": 5.0,
        "min": 5,
        "max": 64,
    }
    assert cell["top_prediction"]["ratio"] == 0.5
    assert cell["unique_prediction_count"] == 2
    assert (tmp_path / "summary.md").read_text() == render_markdown(tmp_path)
    assert all(
        (tmp_path / name).exists()
        for name in [
            "metrics.json",
            "sample_matrix.csv",
            "prediction_distribution.csv",
            "bootstrap_statistics.json",
            "raw_outputs.jsonl",
        ]
    )
    text = (tmp_path / "summary.md").read_text()
    assert all(
        "## " + heading in text
        for heading in [
            "Run Configuration",
            "Data Integrity",
            "Experiment Matrix",
            "Receiver Reasoning Comparison",
            "Upstream Compute Comparison",
            "Combined Metrics",
            "Generation Diagnostics",
            "Prediction Distribution",
            "Sample-level Paired Results",
            "Selected Raw Outputs",
        ]
    )
    assert all(
        "## " + heading not in text
        for heading in ["Conclusion", "Interpretation", "Discussion"]
    )
    saved = json.loads((tmp_path / "metrics.json").read_text())
    saved["derived"]["D_low"]["estimate"] = 0.1234
    (tmp_path / "metrics.json").write_text(json.dumps(saved))
    assert "0.1234" in render_markdown(tmp_path)


def test_mlflow_download_with_no_execution_cache(pilot, tmp_path):
    from mlflow.tracking import MlflowClient

    records, config, _metadata = pilot
    client = MlflowClient(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = client.create_experiment(
        "pilot", artifact_location=str(tmp_path / "store")
    )
    run = client.create_run(experiment, tags={"version_tag": "pilot-v1"})
    run_id = run.info.run_id
    source = tmp_path / "source"
    source.mkdir()
    (source / "sample_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records)
    )
    (source / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    for path in source.iterdir():
        client.log_artifact(run_id, str(path), artifact_path="results")
    client.log_param(run_id, "model_name", config["model_name"])
    client.log_metric(run_id, "substitution_signal", 0)
    client.set_terminated(run_id)
    paths = PathResolver(tmp_path / "fresh_project")
    assert not (paths.root / ".cache").exists()
    rows, loaded_config, loaded_metadata = load_run(client, paths, run_id=run_id)
    assert rows == records and loaded_config == config
    assert loaded_metadata["metrics"]["substitution_signal"] == 0
    assert list(paths.cache_dir.rglob("sample_results.jsonl"))
    assert select_runs(client, version_tag="pilot-v1", experiment_name="pilot") == [
        run_id
    ]
    client.create_run(experiment, tags={"experiment_type": "receiver_reasoning"})
    other = client.create_run(experiment, tags={"version_tag": "pilot-v1"})
    client.log_param(other.info.run_id, "model_name", config["model_name"])
    client.set_terminated(other.info.run_id)
    client.create_run(experiment, tags={"version_tag": "pilot-v1"})
    failed = client.create_run(experiment, tags={"version_tag": "pilot-v1"})
    client.set_terminated(failed.info.run_id, status="FAILED")
    assert set(
        select_runs(client, version_tag="pilot-v1", experiment_name="pilot")
    ) == {run_id, other.info.run_id}
