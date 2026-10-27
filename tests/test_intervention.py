"""Tests for latent intervention use case and tracking metrics."""

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from latentmas_reprop.application.intervention.runner import (
    run_intervention_decoding,
)
from latentmas_reprop.application.intervention_use_case import InterventionUseCase
from latentmas_reprop.domain.models import InterventionMetrics
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
    get_path_resolver,
)
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    load_json,
)


def test_intervention_interrupt_resume_reuse_cpu(tmp_path, monkeypatch):
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    resolver = get_path_resolver()
    run_id = "cpu_interrupt_resume_run"

    model_mock = MagicMock(device=torch.device("cpu"))
    method_mock = MagicMock()
    call_count = 0

    def mock_decode(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("Interrupted during decoding!")
        return [
            {
                "text": f"ans_{call_count}",
                "tokens": [f"ans_{call_count}"],
                "error_msg": None,
                "token_usage": {"completion_tokens": 2},
            }
        ]

    method_mock.decode_with_context.side_effect = mock_decode
    args = MagicMock(
        eval_batch_size=1,
        task="gsm8k",
        model_name="m",
        split="test",
        seed=42,
        latent_steps=0,
    )
    dataset_iter = [{"question": "q0", "gold": "4"}, {"question": "q1", "gold": "10"}]
    sample_keys = ["key_0", "key_1"]
    conditions = ["own", "cross"]
    cross_indices = [1, 0]

    decode_kwargs = {
        "model": model_mock,
        "method": method_mock,
        "args": args,
        "conditions": conditions,
        "cross_indices": cross_indices,
        "dataset_iter": dataset_iter,
        "sample_keys": sample_keys,
        "contexts": [None, None],
        "traces_list": [[], []],
        "context_build_latencies": [0.1, 0.1],
        "context_seq_lens": [10, 10],
        "cache_bytes_list": [100, 100],
        "tracker_port": None,
        "git_hash": "fake",
        "run_id": run_id,
    }

    with pytest.raises(RuntimeError, match="Interrupted during decoding!"):
        run_intervention_decoding(**decode_kwargs)

    c0_own = resolver.get_condition_dir(run_id, "key_0", "own")
    assert is_condition_complete(c0_own)
    res_0 = load_json(c0_own / "result.json")
    assert res_0["sample_key"] == "key_0"
    assert res_0["condition"] == "own"

    method_mock.decode_with_context.reset_mock()
    method_mock.decode_with_context.side_effect = None
    method_mock.decode_with_context.return_value = [
        {
            "text": "resumed_ans",
            "tokens": ["resumed_ans"],
            "error_msg": None,
            "token_usage": {"completion_tokens": 2},
        }
    ]

    records, _, _ = run_intervention_decoding(**decode_kwargs)
    assert method_mock.decode_with_context.call_count == 3
    assert len(records) == 4


def test_intervention_use_case_execute_all_conditions(tmp_path):
    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)

    class _Dataset:
        def __init__(self, samples):
            self._samples = samples

        def load(self, task, split="test", **kwargs):
            return list(self._samples)

    samples = [
        {"question": "What is 2 + 2?", "solution": "4", "gold": "4"},
        {"question": "What is 3 + 3?", "solution": "6", "gold": "6"},
    ]

    dataset_port = _Dataset(samples)
    evaluator_port = MagicMock()
    evaluator_port.evaluate.return_value = True

    tracker = MagicMock()
    tracker.active_run_id = None

    uc = InterventionUseCase(
        cache_port=cache,
        tracker_port=tracker,
        dataset_port=dataset_port,
        evaluator_port=evaluator_port,
    )

    args = _default_args(
        latent_steps=5,
        max_new_tokens=32,
        temperature=0.6,
        cross_pairing_policy="shift_1",
        save_raw_cache=True,
        run_id="integ_run_01",
    )

    model = MagicMock()
    model.model_name = "test_model"
    model.device = torch.device("cpu")
    model.model = MagicMock()
    model.model.dtype = "float32"

    dummy_kv = ((torch.ones((1, 2, 4, 8)), torch.ones((1, 2, 4, 8))),)

    with (
        patch(
            "latentmas_reprop.application.intervention_use_case.LatentMASMethod.build_latent_contexts",
            return_value=(dummy_kv, [[{"trace": "planner"}]]),
        ),
        patch(
            "latentmas_reprop.application.intervention_use_case.LatentMASMethod.decode_with_context",
            return_value=[
                {
                    "prediction": "4",
                    "raw_prediction": "The answer is 4",
                    "error_msg": None,
                    "generated_tokens": 5,
                    "judger_trace": {"step": 1},
                }
            ],
        ) as mock_decode,
    ):
        _metrics, records = uc.execute(model=model, args=args)
        assert len(records) == 8
        assert mock_decode.call_count == 8

        mock_decode.reset_mock()
        _metrics2, records2 = uc.execute(model=model, args=args)
        assert len(records2) == 8
        assert mock_decode.call_count == 0


def _default_args(**kwargs):
    defaults = {
        "method": "latent_mas",
        "task": "gsm8k",
        "split": "test",
        "max_samples": 2,
        "latent_steps": 4,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.95,
        "seed": 42,
        "model_name": "test-model",
        "tracking_experiment_name": "test_exp",
        "prompt": "sequential",
        "use_vllm": False,
        "save_raw_cache": False,
        "cross_policy": "shift_1",
        "intervention_conditions": "own,cross,drop,zero",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_mlflow_run_created_before_execution():
    call_order: list[str] = []
    tracker = MagicMock()
    tracker.start_run.side_effect = lambda **_kw: call_order.append("start_run")
    tracker.log_params.return_value = None
    tracker.start_sample_trace.return_value.__enter__ = lambda s: (
        call_order.append("trace_enter") or MagicMock()
    )
    tracker.start_sample_trace.return_value.__exit__ = lambda s, *a: False

    uc = InterventionUseCase(tracker_port=tracker)
    dataset_mock = MagicMock(
        load=lambda **kw: [
            {"question": "q0", "gold": "1"},
            {"question": "q1", "gold": "2"},
        ]
    )
    args = _default_args()
    model_mock = MagicMock(
        device=torch.device("cpu"), model=MagicMock(dtype="bfloat16")
    )

    def fake_run_experiment(**_kw):
        call_order.append("run_experiment")
        return (InterventionMetrics(n_samples=0), [])

    with (
        patch.object(uc, "dataset_port", dataset_mock),
        patch.object(uc, "_run_experiment", fake_run_experiment),
        contextlib.suppress(Exception),
    ):
        uc.execute(model=model_mock, args=args)

    assert call_order.index("start_run") < call_order.index("run_experiment")


def test_failed_experiment_calls_end_run_failed():
    tracker = MagicMock(start_run=lambda *a, **kw: None, log_params=lambda *a, **kw: None)
    uc = InterventionUseCase(tracker_port=tracker)
    dataset_mock = MagicMock(
        load=lambda **kw: [{"question": "q0", "gold": "1"}, {"question": "q1", "gold": "2"}]
    )
    args = _default_args()
    model_mock = MagicMock(
        device=torch.device("cpu"), model=MagicMock(dtype="bfloat16")
    )

    def boom(**_kw):
        raise RuntimeError("Simulated OOM")

    with (
        patch.object(uc, "dataset_port", dataset_mock),
        patch.object(uc, "_run_experiment", boom),
        pytest.raises(RuntimeError, match="Simulated OOM"),
    ):
        uc.execute(model=model_mock, args=args)

    tracker.end_run.assert_called_once_with(status="FAILED")


def test_intervention_context_builder_saves_shared_kv(tmp_path):
    from latentmas_reprop.application.intervention.context_builder import (
        build_intervention_contexts,
    )

    resolver = PathResolver(tmp_path)
    cache = ExecutionCacheManager(resolver)
    method_mock = MagicMock(
        latent_steps=5,
        model=MagicMock(model_name="test_model"),
        build_latent_contexts=lambda *a, **k: (
            ((torch.zeros((1, 2, 4, 8)), torch.zeros((1, 2, 4, 8))),),
            [[{"trace": "d"}]],
        ),
    )
    run_id = "test_intervention_run"
    build_intervention_contexts(
        method=method_mock,
        dataset_iter=[{"question": "2+2?"}],
        task="gsm8k",
        seed=42,
        save_raw_cache=False,
        cache_port=cache,
        split="test",
        run_id=run_id,
    )
    shared_dir = resolver.get_condition_dir(run_id, 0, "shared")
    assert (shared_dir / "kv.pt").is_file()
    assert (shared_dir / "result.json").is_file()
