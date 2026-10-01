"""Preflight pairing, invalid-free handling, and trajectory execution contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmas_reprop.application.receiver_compute_preflight.analysis import (
    analyze,
    export,
)
from latentmas_reprop.application.receiver_compute_preflight.inference import (
    length_matched_donors,
)
from latentmas_reprop.infrastructure.models.text_generation import (
    generate_token_ids_batch,
)
from src.run.cli import parse_run_matrix


def config_and_records():
    config = {
        "upstream_steps": [10, 20],
        "receiver_budgets": [2, 4, "free"],
        "target_accuracies": [0.5, 1],
        "bootstrap_count": 100,
        "seed": 7,
        "sample_ids": ["a", "b"],
    }
    rows = []
    for u in config["upstream_steps"]:
        for m in ("matched", "mismatched", "no_handoff"):
            for b in config["receiver_budgets"]:
                for sid in config["sample_ids"]:
                    rows.append(
                        {
                            "upstream_steps": u,
                            "handoff_condition": m,
                            "receiver_budget": b,
                            "sample_id": sid,
                            "valid": True,
                            "correct": m == "matched" and (b != 2 or sid == "a"),
                        }
                    )
    return config, rows


def test_paired_bootstrap_and_thresholds(tmp_path):
    config, rows = config_and_records()
    metrics, statistics = analyze(rows, config)
    gain = metrics["comparisons"][0]
    assert gain["difference"] == 0.5
    indices = np.array(statistics["resample_indices"])
    assert (
        statistics["distributions"]["10:pairing_gain:2"]
        == np.array([1, 0])[indices].mean(axis=1).tolist()
    )
    assert metrics["thresholds"][0]["minimum_receiver_budget"] == 2
    assert metrics["thresholds"][1]["minimum_receiver_budget"] is None
    assert (
        next(
            r
            for r in metrics["thresholds"]
            if r["target_accuracy"] == 1 and r["handoff_condition"] == "matched"
        )["minimum_receiver_budget"]
        == 4
    )
    assert analyze(rows[::-1], config) == (metrics, statistics)
    export(tmp_path, rows, config)
    assert {p.name for p in tmp_path.iterdir()} == {
        "metrics.json",
        "bootstrap_statistics.json",
        "summary.md",
        "sample_matrix.csv",
        "budget_curves.csv",
    }


def test_missing_duplicate_or_changed_sample_sets_refuse_pairing():
    config, rows = config_and_records()
    with pytest.raises(ValueError, match="mismatch"):
        analyze(rows[:-1], config)
    with pytest.raises(ValueError, match="Duplicate"):
        analyze([*rows, rows[0]], config)
    changed = deepcopy(rows)
    changed[0]["sample_id"] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        analyze(changed, config)


def test_invalid_free_has_no_accuracy_or_paired_gain():
    config, rows = config_and_records()
    for row in rows:
        if row["receiver_budget"] == "free" and row["handoff_condition"] == "matched":
            row["valid"] = False
    metrics, _ = analyze(rows, config)
    assert all(
        r["accuracy"] is None
        for r in metrics["curves"]
        if r["receiver_budget"] == "free" and r["handoff_condition"] == "matched"
    )
    assert all(
        r["difference"] is None
        for r in metrics["comparisons"]
        if r["receiver_budget"] == "free"
        and r["comparison"] != "mismatched_vs_no_handoff"
    )


def test_length_derangement_is_bijective_deterministic():
    lengths = [10, 5, 10, 40, 6]
    donors = length_matched_donors(lengths)
    assert donors == length_matched_donors(lengths)
    assert sorted(donors) == list(range(5))
    assert all(i != d for i, d in enumerate(donors))
    with pytest.raises(ValueError):
        length_matched_donors([1])


def test_exact_generated_ids_include_eos_even_when_pad_equals_eos():
    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert "temperature" not in kwargs
            return SimpleNamespace(
                sequences=torch.tensor([[9, 3, 2]]), past_key_values=None
            )

    ids, _ = generate_token_ids_batch(
        Model(),
        SimpleNamespace(pad_token_id=2),
        torch.device("cpu"),
        torch.tensor([[9]]),
        temperature=0,
        max_new_tokens=2,
    )
    assert ids == [[3, 2]]


def test_cli_axes_and_constraints():
    base = ["-c", "lmas/receiver_compute_preflight/gsm8k", "--dry-run"]
    bare = parse_run_matrix(["--receiver_compute_preflight", "--dry-run"])[0]
    assert bare.upstream_steps == [10, 20]
    assert bare.max_new_tokens == 4096
    assert bare.model_name == "Qwen/Qwen3-4B"
    args = parse_run_matrix(base)[0]
    assert args.upstream_steps == [10, 20]
    assert args.receiver_budgets == [64, 128, 256, 512, 1024, "free"]
    assert (args.temperature, args.top_p) == (0, 1)
    for flags in (
        ["--max_samples", "1"],
        ["--handoff_positions", "5"],
        ["--receiver_reasoning"],
        ["--receiver_budgets", "4,4,free"],
        ["--latent_only"],
        ["--verify_prefix"],
    ):
        with pytest.raises(SystemExit):
            parse_run_matrix([*base, *flags])


def test_collection_builds_each_context_once_and_uses_other_sample(monkeypatch):
    import latentmas_reprop.application.receiver_compute_preflight_use_case as module

    class Tracker:
        def __init__(self):
            self.artifacts = {}
            self.status = None

        def start_run(self, **kwargs):
            pass

        def log_params(self, config):
            self.config = config

        def log_artifact(self, path, **kwargs):
            self.artifacts[path.name] = path.read_text()

        def log_metrics(self, metrics):
            pass

        def flush_traces(self):
            pass

        def end_run(self, status):
            self.status = status

    class Dataset:
        def load(self, **kwargs):
            return [
                {"question": "first", "gold": "1"},
                {"question": "second", "gold": "2"},
            ]

    tracker = Tracker()
    args = parse_run_matrix(
        [
            "-c",
            "lmas/receiver_compute_preflight/gsm8k",
            "--max_samples",
            "2",
            "--receiver_budgets",
            "2,4,free",
            "--max_new_tokens",
            "8",
            "--bootstrap_count",
            "10",
        ]
    )[0]
    calls = []

    def build(method, item, step, width, context_id, tracker, runtime):
        assert width is None
        calls.append((item["question"], step))
        return (
            None,
            [[{"latent_steps": step}]],
            {"handoff_positions": len(item["question"])},
        )

    def generate(method, item, context, limit, report_progress=False):
        return [1, 2][:limit], {"receiver_input_ids": [5]}

    monkeypatch.setattr(module, "build_upstream", build)
    monkeypatch.setattr(module, "generate_receiver", generate)
    model = SimpleNamespace(
        device="cpu",
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids[-1])),
    )
    metrics, rows = module.ReceiverComputePreflightUseCase(
        dataset_port=Dataset(), tracker_port=tracker
    ).execute(model, args)
    assert len(calls) == 4
    assert len(rows) == 36
    assert metrics["sample_count"] == 2
    assert tracker.status == "FINISHED"
    assert "sample_results.jsonl" in tracker.artifacts
    assert all(
        r["sample_id"] != r["donor_id"]
        for r in rows
        if r["handoff_condition"] == "mismatched"
    )
    assert all(r["valid"] for r in rows)
