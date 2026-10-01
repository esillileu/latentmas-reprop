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
    result = collect_trajectory(
        method, {"gold": "42"}, context, args, DEFAULT_EVALUATOR
    )
    assert calls == [(8, context), (16, context)]
    assert result["free_naturally_terminated"] is True
    assert result["pathological"] is False
    assert result["free_initial_cap_reached"] is True
    assert result["free_retry_count"] == 1
    assert result["receiver_retry_latency_sec"] == 0.8
    assert result["metadata"]["receiver_latency_sec"] == 1.6


def test_natural_eos_at_cap_is_valid_and_does_not_retry(monkeypatch):
    method, args = method_and_args()
    calls = generator(monkeypatch, stop_at=8)
    result = collect_trajectory(method, {"gold": "42"}, None, args, DEFAULT_EVALUATOR)
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
    result = collect_trajectory(method, {"gold": "42"}, None, args, DEFAULT_EVALUATOR)
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
    assert all(
        r["donor_length_delta"] == 2 and r["donor_length_abs_delta"] == 2 for r in rows
    )
    assert rows[-1]["free_cap_reached"] is True


def test_expansion_refuses_changed_prefix(monkeypatch):
    method, args = method_and_args()
    generator(monkeypatch, stop_at=12, mutate=True)
    with pytest.raises(ValueError, match="initial greedy prefix"):
        collect_trajectory(method, {"gold": "42"}, None, args, DEFAULT_EVALUATOR)


def test_actual_capped_smoke_verification_ignores_timing_variation(monkeypatch):
    method, args = method_and_args()
    args.verify_prefix = True
    calls = generator(monkeypatch, stop_at=6)
    result = collect_trajectory(method, {"gold": "42"}, None, args, DEFAULT_EVALUATOR)
    assert [c[0] for c in calls] == [8, 2, 4]
    assert result["prefix_verified"] is True
