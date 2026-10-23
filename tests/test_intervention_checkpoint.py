"""Tests for intervention checkpoint resumption."""

import json
from unittest.mock import MagicMock

import torch

from latentmas_reprop.application.intervention.runner import (
    run_intervention_decoding,
)
from latentmas_reprop.domain.models import SampleInterventionRecord


def test_intervention_checkpoint_resumption(tmp_path):
    """Verify that interrupted intervention runs resume from checkpoint without re-evaluating."""
    model_mock = MagicMock()
    model_mock.device = torch.device("cpu")
    method_mock = MagicMock()

    chk_file = tmp_path / "intervention_checkpoint_test.jsonl"
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
    chk_file.write_text(json.dumps(sample_rec.to_dict()) + "\n", encoding="utf-8")

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
        incremental_jsonl_path=chk_file,
    )

    assert len(records) == 2
    # decode_with_context called only once for key_1
    assert method_mock.decode_with_context.call_count == 1


def test_intervention_directory_resumption(tmp_path, monkeypatch):
    """Verify that interrupted intervention runs resume from per-condition directory without re-evaluating."""
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    from latentmas_reprop.infrastructure.paths.resolver import get_path_resolver
    from latentmas_reprop.infrastructure.storage import save_condition_result

    resolver = get_path_resolver()
    run_id = "test_run_dir_resumption"

    # Pre-populate sample key_0 condition 'own'
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
    # decode_with_context called only once for key_1 because key_0 was resumed from directory
    assert method_mock.decode_with_context.call_count == 1
