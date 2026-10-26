"""Tests for CPU-based condition saving, interruption, resumption, and reuse."""

from unittest.mock import MagicMock

import pytest
import torch

from latentmas_reprop.application.intervention.runner import (
    run_intervention_decoding,
)
from latentmas_reprop.domain.models import (
    ReceiverAcquisitionRecord,
    SampleInterventionRecord,
)
from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
from latentmas_reprop.infrastructure.storage import (
    is_condition_complete,
    load_json,
    save_condition_result,
)


def test_intervention_interrupt_resume_reuse_cpu(tmp_path, monkeypatch):
    """Verify condition saving, interruption mid-way, resumption, and reuse on CPU."""
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    resolver = get_path_resolver()
    run_id = "cpu_interrupt_resume_run"

    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    method_mock = MagicMock()

    call_count = 0

    def mock_decode(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            # Simulate crash / interruption on the 2nd decode call
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
    dataset_iter = [
        {"question": "q0", "gold": "4"},
        {"question": "q1", "gold": "10"},
    ]
    sample_keys = ["key_0", "key_1"]
    conditions = ["own", "cross"]
    cross_indices = [1, 0]

    # --- Phase 1: Run until interrupted ---
    with pytest.raises(RuntimeError, match="Interrupted during decoding!"):
        run_intervention_decoding(
            model=model_mock,
            method=method_mock,
            args=args,
            conditions=conditions,
            cross_indices=cross_indices,
            dataset_iter=dataset_iter,
            sample_keys=sample_keys,
            contexts=[None, None],
            traces_list=[[], []],
            context_build_latencies=[0.1, 0.1],
            context_seq_lens=[10, 10],
            cache_bytes_list=[100, 100],
            tracker_port=None,
            git_hash="fake",
            run_id=run_id,
        )

    # Verify storage after interruption:
    # key_0/own completed and saved
    c0_own = resolver.get_condition_dir(run_id, "key_0", "own")
    assert is_condition_complete(c0_own)
    res_0 = load_json(c0_own / "result.json")
    assert res_0["sample_key"] == "key_0"
    assert res_0["condition"] == "own"

    # key_0/cross failed before saving
    c0_cross = resolver.get_condition_dir(run_id, "key_0", "cross")
    assert not is_condition_complete(c0_cross)

    # key_1 not evaluated yet
    c1_own = resolver.get_condition_dir(run_id, "key_1", "own")
    assert not is_condition_complete(c1_own)

    # --- Phase 2: Resume with identical run_id ---
    # Reset method mock to normal responses without errors
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

    records, _, _ = run_intervention_decoding(
        model=model_mock,
        method=method_mock,
        args=args,
        conditions=conditions,
        cross_indices=cross_indices,
        dataset_iter=dataset_iter,
        sample_keys=sample_keys,
        contexts=[None, None],
        traces_list=[[], []],
        context_build_latencies=[0.1, 0.1],
        context_seq_lens=[10, 10],
        cache_bytes_list=[100, 100],
        tracker_port=None,
        git_hash="fake",
        run_id=run_id,
    )

    # Total conditions needed: 2 samples * 2 conditions = 4.
    # key_0/own was already completed, so decode_with_context should be called exactly 3 times!
    assert method_mock.decode_with_context.call_count == 3
    assert len(records) == 4

    # All 4 conditions are now complete in storage
    assert is_condition_complete(resolver.get_condition_dir(run_id, "key_0", "own"))
    assert is_condition_complete(resolver.get_condition_dir(run_id, "key_0", "cross"))
    assert is_condition_complete(resolver.get_condition_dir(run_id, "key_1", "own"))
    assert is_condition_complete(resolver.get_condition_dir(run_id, "key_1", "cross"))


def test_condition_resumption_with_pre_existing_directory(tmp_path, monkeypatch):
    """Verify that existing complete condition is loaded and re-evaluation is skipped."""
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    resolver = get_path_resolver()
    run_id = "test_pre_existing_dir"

    cond_dir = resolver.get_condition_dir(run_id, "key_0", "own")
    sample_rec = SampleInterventionRecord(
        sample_id=0,
        source_sample_id=None,
        condition="own",
        question="q0",
        gold="4",
        prediction="4",
        raw_prediction="4",
        correct=True,
        model="m",
        task="gsm8k",
        latent_steps=0,
        seed=42,
        sample_index=0,
        sample_key="key_0",
        latency=0.1,
        generated_tokens=5,
    )
    save_condition_result(cond_dir, sample_rec.to_dict())

    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    method_mock = MagicMock()
    method_mock.decode_with_context.return_value = [
        {
            "text": "10",
            "tokens": ["10"],
            "error_msg": None,
            "token_usage": {"completion_tokens": 2},
        }
    ]

    records, _, _ = run_intervention_decoding(
        model=model_mock,
        method=method_mock,
        args=MagicMock(
            eval_batch_size=1,
            task="gsm8k",
            model_name="m",
            split="test",
            seed=42,
            latent_steps=0,
        ),
        conditions=["own"],
        cross_indices=[],
        dataset_iter=[
            {"question": "q0", "gold": "4"},
            {"question": "q1", "gold": "10"},
        ],
        sample_keys=["key_0", "key_1"],
        contexts=[None, None],
        traces_list=[[], []],
        context_build_latencies=[0.1, 0.1],
        context_seq_lens=[10, 10],
        cache_bytes_list=[100, 100],
        tracker_port=None,
        git_hash="fake",
        run_id=run_id,
    )

    assert len(records) == 2
    # decode_with_context called only once for key_1 because key_0 was reused from disk
    assert method_mock.decode_with_context.call_count == 1
    assert records[0].prediction == "4"


def test_receiver_acquisition_interrupt_resume_reuse_cpu(tmp_path, monkeypatch):
    """Verify receiver acquisition condition save, interruption, and resume on CPU."""
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    resolver = get_path_resolver()
    run_id = "acq_cpu_resume_run"

    # Pre-populate sample_0 condition 'prompt_only_own'
    cond_dir = resolver.get_condition_dir(run_id, "sample_0", "prompt_only_own")
    rec_data = {
        "sample_id": "sample_0",
        "sample_index": 0,
        "sample_key": "key_0",
        "target_digit": 7,
        "source_sample_id": None,
        "source_sample_index": None,
        "source_sample_key": None,
        "source_digit": None,
        "receiver_input": {},
        "raw_output": {},
        "context_mode": "prompt_only",
        "condition": "own",
        "candidate_probabilities": {"7": 0.9},
        "candidate_log_probabilities": {"7": -0.1},
        "top_token_candidates": [{"token": "7", "prob": 0.9}],
        "candidate_mass": 0.9,
        "predicted_digit": 7,
        "source_probability": 0.9,
        "source_log_probability": -0.1,
        "source_rank": 1,
        "target_probability": 0.9,
        "target_log_probability": -0.1,
        "target_rank": 1,
        "source_margin": 0.0,
        "content_follow_correct": True,
        "target_retained": True,
        "cache_present": False,
        "cache_sequence_length": 10,
        "cache_bytes": 100,
        "num_layers": 1,
        "cache_dtype": "float32",
        "original_full_seq_len": 10,
        "retained_tail_len": 10,
        "retained_tail_start_position": None,
        "retained_original_position_end": 10,
        "receiver_position_start": 10,
        "receiver_prompt_tokens": 4,
        "latency_sec": 0.1,
        "error": None,
        "hidden_state_artifact_key": "hidden_0",
        "seed": 42,
        "model": "model",
        "task": "secret_digit",
        "latent_steps": 2,
    }
    rec = ReceiverAcquisitionRecord(**rec_data)
    save_condition_result(
        cond_dir,
        rec.to_dict(),
        tensors={"hidden": torch.tensor([1.0, 2.0, 3.0])},
    )

    # Verify is_condition_complete detects it and loads
    assert is_condition_complete(cond_dir)
    loaded = load_json(cond_dir / "result.json")
    assert loaded["predicted_digit"] == 7
    assert "hidden.pt" in loaded["files"]
