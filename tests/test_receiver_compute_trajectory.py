"""Only truncated free trajectories retry; unresolved ones remain pathological."""

from types import SimpleNamespace

import pytest

from latentmas_reprop.application.receiver_compute_preflight.trajectory import (
    collect_trajectory,
    termination,
)
from latentmas_reprop.infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR


def method_and_args():
    tokenizer = SimpleNamespace(
        eos_token_id=99, decode=lambda ids, **kwargs: r"\boxed{42}"
    )
    method = SimpleNamespace(
        model=SimpleNamespace(
            tokenizer=tokenizer,
            model=SimpleNamespace(
                generation_config=SimpleNamespace(eos_token_id=[99, 100])
            ),
        )
    )
    args = SimpleNamespace(
        seed=0,
        receiver_budgets=[2, 4, "free"],
        max_new_tokens=8,
        free_max_new_tokens=32,
        verify_prefix=False,
    )
    return method, args


def generator(monkeypatch, stop_at=None, mutate=False):
    import latentmas_reprop.application.receiver_compute_preflight.trajectory as module

    calls = []

    def generate(method, item, context, cap, budgets=(), report_progress=False):
        calls.append((cap, context))
        ids = list(range(min(cap, stop_at or cap)))
        if stop_at and cap >= stop_at:
            ids[-1] = 99
        if mutate and len(calls) > 1:
            ids[0] = -1
        return ids, {
            "receiver_latency_sec": cap / 10,
            "receiver_budget_latency_sec": {str(b): b / 10 for b in budgets},
            "receiver_prompt": "prompt",
            "receiver_input_ids": [5],
            "receiver_thinking_open": False,
            "receiver_cache_positions": 100,
            "receiver_prompt_tokens": 1,
        }

    monkeypatch.setattr(module, "generate_receiver", generate)
    return calls


def test_only_cap_reaching_trajectory_retries_and_preserves_context(monkeypatch):
    method, args = method_and_args()
    calls = generator(monkeypatch, stop_at=12)
    context = object()
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, context, args)
    assert calls == [(8, context), (16, context)]
    assert result["free_naturally_terminated"] is True
    assert result["pathological"] is False
    assert result["free_initial_cap_reached"] is True
    assert len(result["free_attempts"]) == 2
    assert "free_retry_count" not in result
    assert "receiver_retry_latency_sec" not in result
    assert result["metadata"]["receiver_latency_sec"] == 1.6


def test_natural_eos_at_cap_is_valid_and_does_not_retry(monkeypatch):
    method, args = method_and_args()
    calls = generator(monkeypatch, stop_at=8)
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)
    assert len(calls) == 1
    assert result["free_naturally_terminated"] is True
    assert result["pathological"] is False
    assert termination(method.model, [1, 100], 2) == "eos"


def test_retry_ceiling_produces_explicit_pathological_records(monkeypatch):
    from latentmas_reprop.application.receiver_compute_preflight.records import (
        trajectory_records,
    )

    method, args = method_and_args()
    args.model_name, args.seed = "model", 0
    calls = generator(monkeypatch)
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)
    assert [c[0] for c in calls] == [8, 16, 32]
    assert result["pathological"] is True
    rows = trajectory_records(
        method.model,
        DEFAULT_EVALUATOR,
        {"question": "q", "gold": "42"},
        args,
        result,
        {"donor_id": "other"},
        {"handoff_positions": 98},
        {"handoff_positions": 100},
        [],
    )
    assert [r["valid"] for r in rows] == [True, True, False]
    from latentmas_reprop.application.receiver_compute_preflight.answers import (
        reevaluate_records,
    )

    evaluated = reevaluate_records(rows, DEFAULT_EVALUATOR)
    assert all(
        r["donor_length_delta"] == 2 and r["donor_length_abs_delta"] == 2
        for r in evaluated
    )
    assert rows[-1]["free_cap_reached"] is True


def test_expansion_refuses_changed_prefix(monkeypatch):
    method, args = method_and_args()
    generator(monkeypatch, stop_at=12, mutate=True)
    with pytest.raises(ValueError, match="initial greedy prefix"):
        collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)


def test_actual_capped_smoke_verification_ignores_timing_variation(monkeypatch):
    method, args = method_and_args()
    args.verify_prefix = True
    calls = generator(monkeypatch, stop_at=6)
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)
    assert [c[0] for c in calls] == [8, 2, 4]
    from latentmas_reprop.application.receiver_compute_preflight.answers import (
        trajectory_diagnostics,
    )

    row = result | result["metadata"]
    assert "prefix_verified" not in result
    assert trajectory_diagnostics(row)["prefix_verified"] is True


@pytest.mark.parametrize("changed", ["ids", "prompt"])
def test_collection_refuses_capped_mismatch_before_success(monkeypatch, changed):
    import latentmas_reprop.application.receiver_compute_preflight.trajectory as module

    method, args = method_and_args()
    args.verify_prefix = True
    generator(monkeypatch, stop_at=6)
    original = module.generate_receiver

    def generate(*positional, **kwargs):
        ids, metadata = original(*positional, **kwargs)
        if positional[3] == 2:
            if changed == "ids":
                ids[0] = -1
            else:
                metadata["receiver_prompt"] = "different prompt"
        return ids, metadata

    monkeypatch.setattr(module, "generate_receiver", generate)
    with pytest.raises(ValueError, match="Greedy prefix differs"):
        collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)


def test_reanalysis_checks_original_prefix_ids_and_requires_all_capped_checks(
    monkeypatch,
):
    from copy import deepcopy

    from latentmas_reprop.application.receiver_compute_preflight.answers import (
        trajectory_diagnostics,
    )

    method, args = method_and_args()
    args.verify_prefix = True
    generator(monkeypatch, stop_at=6)
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)
    row = (
        result
        | result["metadata"]
        | {
            "receiver_budget": 2,
            "generated_token_ids": result["token_ids"][:2],
            "generated_tokens": 2,
        }
    )
    assert trajectory_diagnostics(row)["prefix_verified"] is True
    corrupted = deepcopy(row)
    corrupted["generated_token_ids"][0] = -1
    with pytest.raises(ValueError, match="Budget record differs"):
        trajectory_diagnostics(corrupted)
    missing = deepcopy(row)
    missing["prefix_verification_attempts"].pop()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        trajectory_diagnostics(missing)


def test_collection_compares_strict_capped_scoring(monkeypatch):
    from latentmas_reprop.application.receiver_compute_preflight.records import (
        trajectory_records,
    )

    method, args = method_and_args()
    args.verify_prefix = True
    args.model_name = "model"
    generator(monkeypatch, stop_at=6)
    result = collect_trajectory(method, {"question": "q", "gold": "42"}, None, args)
    result["prefix_verification_attempts"][0]["raw_receiver_output"] = (
        "Intermediate: 42"
    )
    with pytest.raises(ValueError, match="strict prefix evaluation"):
        trajectory_records(
            method.model,
            DEFAULT_EVALUATOR,
            {"question": "q", "gold": "42"},
            args,
            result,
            {"donor_id": None},
            {"handoff_positions": 0},
            {},
            [],
        )


def test_prefix_failure_marks_mlflow_run_failed_and_uploads_evidence(
    monkeypatch, tmp_path
):
    import json

    from mlflow.tracking import MlflowClient

    import latentmas_reprop.application.receiver_compute_preflight_use_case as module
    from latentmas_reprop.infrastructure.tracking.mlflow_tracker import MLflowTracker
    from src.run.cli import parse_run_matrix

    method, trajectory_args = method_and_args()
    trajectory_args.verify_prefix = True
    generator(monkeypatch, stop_at=6)

    def collect(*args, **kwargs):
        from latentmas_reprop.application.receiver_compute_preflight import trajectory

        original = trajectory.generate_receiver

        def corrupt(*positional, **kw):
            ids, metadata = original(*positional, **kw)
            if positional[3] == 2:
                ids[0] = -1
            return ids, metadata

        monkeypatch.setattr(trajectory, "generate_receiver", corrupt)
        collect_trajectory(
            method, {"question": "q", "gold": "42"}, None, trajectory_args
        )

    monkeypatch.setattr(module, "collect_samples", collect)
    tracker = MLflowTracker(
        tracking_uri=f"sqlite:///{tmp_path / 'runs.db'}",
        artifact_location=str(tmp_path / "artifacts"),
    )
    args = parse_run_matrix(["--receiver_compute_preflight", "--max_samples", "2"])[0]
    dataset = SimpleNamespace(load=lambda **kw: [{"question": "a"}, {"question": "b"}])
    with pytest.raises(ValueError, match="Greedy prefix differs"):
        module.ReceiverComputePreflightUseCase(
            dataset_port=dataset, tracker_port=tracker
        ).execute(SimpleNamespace(device="cpu"), args)
    client = MlflowClient()
    run = client.search_runs(
        [client.get_experiment_by_name(args.tracking_experiment_name).experiment_id]
    )[0]
    assert run.info.status == "FAILED"
    failure = client.download_artifacts(
        run.info.run_id, "results/failure.json", str(tmp_path)
    )
    with open(failure) as f:
        assert "Greedy prefix differs" in json.load(f)["traceback"]
