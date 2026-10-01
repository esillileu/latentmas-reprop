import json
from types import SimpleNamespace

import pytest

from latentmas_reprop.application.receiver_reasoning.collection import analyze_runs
from latentmas_reprop.application.receiver_reasoning.mlflow import select_runs


def candidate(model, run_id, version="pilot-v1", status="FINISHED", commit="abc"):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status),
        data=SimpleNamespace(
            params={"model_name": model},
            tags={"version_tag": version, "git_commit": commit},
        ),
    )


def test_finished_version_runs_include_all_models_and_commits():
    runs = [
        candidate("Qwen/Qwen3-4B", "4"),
        candidate("Qwen/Qwen3-8B", "8", status="FAILED", commit="other"),
        candidate("Qwen/Qwen3-14B", "14", status="RUNNING"),
        candidate("Qwen/Qwen3-4B", "4other", commit="other"),
        candidate("Qwen/Qwen3-4B", "excluded", version="pilot-v2"),
        candidate("Qwen/Qwen3-4B", "untagged", version=None),
    ]

    def search_runs(*args, **kwargs):
        assert kwargs["filter_string"] == "attributes.status = 'FINISHED'"
        return [run for run in runs if run.info.status == "FINISHED"]

    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="1"),
        search_runs=search_runs,
    )
    assert set(select_runs(client, version_tag="pilot-v1")) == {
        "4",
        "4other",
    }
    assert select_runs(client, version_tag="pilot-v2") == ["excluded"]
    with pytest.raises(ValueError, match="No FINISHED MLflow runs"):
        select_runs(client, version_tag="missing")
    with pytest.raises(ValueError, match="nonempty"):
        select_runs(client, version_tag=" ")


def test_version_selection_checks_all_pages():
    class Page(list):
        def __init__(self, rows, token):
            super().__init__(rows)
            self.token = token

    def search_runs(*args, page_token=None, **kwargs):
        return (
            Page([candidate("4B", "4")], "next")
            if page_token is None
            else Page([candidate("8B", "8")], None)
        )

    client = SimpleNamespace(
        get_experiment_by_name=lambda name: SimpleNamespace(experiment_id="1"),
        search_runs=search_runs,
    )
    assert select_runs(client, version_tag="pilot-v1") == ["4", "8"]


def test_three_model_reports_remain_independent(monkeypatch, tmp_path):
    def load_run(client, paths, *, run_id):
        model = f"Qwen/Qwen3-{run_id}B"
        config = {
            "model_name": model,
            "task": "gsm8k",
            "split": "test",
            "prompt": "sequential",
            "seed": 42,
            "temperature": 0.6,
            "top_p": 0.95,
            "upstream_steps": [10, 40],
            "max_new_tokens": 2048,
            "answer_only_max_new_tokens": 64,
        }
        records = [
            {
                "sample_id": "same_sample",
                "sample_index": 0,
                "gold": "1",
                "question": "q",
                "upstream_latent_steps": step,
                "receiver_mode": mode,
                "prediction": "1",
                "correct": run_id == "4",
                "raw_receiver_output": "1",
                "receiver_generated_tokens": 1,
                "context_id": f"0:{step}",
                "upstream_context_sequence_length": 100,
                "model": model,
                "seed": 42,
                "config": config,
            }
            for step in (10, 40)
            for mode in ("answer_only", "free")
        ]
        metadata = {
            "run_id": run_id,
            "experiment_id": "1",
            "status": "FINISHED",
            "start_time": 0,
            "end_time": 1,
            "artifact_uri": "uri",
            "params": {},
            "metrics": {},
            "tags": {},
        }
        return records, config, metadata

    monkeypatch.setattr(
        "latentmas_reprop.application.receiver_reasoning.collection.load_run", load_run
    )
    entries = analyze_runs(None, None, ["4", "8", "14"], tmp_path, bootstrap_count=100)
    assert len(entries) == 3
    for entry in entries:
        report = tmp_path / entry["report"]
        assert report.exists()
        metrics = json.loads((report.parent / "metrics.json").read_text())
        assert metrics["config"]["model_name"] == entry["model"]
        assert metrics["cells"]["low_free"]["sample_count"] == 1
        assert metrics["cells"]["low_free"]["accuracy"] == (
            1 if entry["run_id"] == "4" else 0
        )
        assert report.parent.joinpath("bootstrap_statistics.json").exists()
    index = tmp_path.joinpath("summary.md").read_text()
    assert all(entry["report"] in index for entry in entries)
    assert "not pooled" in index


@pytest.mark.parametrize("option", ["--run-id", "--model-name", "--git-commit"])
def test_other_selection_options_are_rejected(option):
    from src.run.handoff_analysis import main

    with pytest.raises(SystemExit):
        main(["--version-tag", "pilot-v1", option, "value"])
