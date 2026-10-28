"""Tests for intermediate latent step checkpointing and incremental resumption."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from latentmas_reprop.application.single_intervention.checkpoint import (
    find_latent_step_checkpoint,
    make_step_checkpoint_saver,
)
from latentmas_reprop.application.single_intervention.runner import (
    SingleInterventionRunner,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.device = torch.device("cpu")
    model.tokenizer = MagicMock()
    model.tokenizer.pad_token_id = 0
    model.tokenizer.eos_token_id = 2
    model.tokenizer.convert_ids_to_tokens.side_effect = lambda ids: [
        f"tok_{i}" for i in ids
    ]
    model.tokenizer.decode.return_value = "Mock answer 42"
    model.prepare_chat_batch.return_value = (
        ["mock prompt"],
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[1, 1, 1]], dtype=torch.long),
        [["mock", "prompt"]],
    )
    dummy_out = MagicMock()
    dummy_out.past_key_values = ((torch.zeros(1, 1, 1, 4), torch.zeros(1, 1, 1, 4)),)
    dummy_out.hidden_states = [torch.zeros(1, 3, 4)]
    dummy_head = MagicMock()
    dummy_head.return_value = torch.tensor([[0.1, 0.2, 0.9]])
    model.model = MagicMock(return_value=dummy_out)
    model.model.lm_head = dummy_head
    model._apply_latent_realignment.return_value = torch.zeros(4)
    return model


def test_find_and_save_latent_step_checkpoint(tmp_path):
    storage = FileStorage(PathResolver(tmp_path))
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/Qwen3-8B",
        method="latent_mas",
        prompt="sequential",
        seed=42,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=2048,
        single_intervention=True,
        latent_steps=20,
    )
    cb = make_step_checkpoint_saver(
        storage=storage,
        args=args,
        dtype_name="float16",
        backend="transformers",
        skey="000000",
        idx=0,
        overwrite=False,
    )
    kv_tensor = ((torch.ones(1, 1, 1, 4), torch.ones(1, 1, 1, 4)),)
    hidden_tensor = torch.ones(15, 4)
    meta = {"prompt_length": 5, "latent_steps": 10}

    # Save intermediate checkpoint at step 10
    cb(10, kv_tensor, hidden_tensor, meta)

    # When searching for checkpoints < 20 with interval 10, should find step 10
    found_step, found_kv, found_hidden, found_meta = find_latent_step_checkpoint(
        storage=storage,
        args=args,
        dtype_name="float16",
        backend="transformers",
        skey="000000",
        target_steps=20,
        interval=10,
    )
    assert found_step == 10
    assert found_kv is not None
    assert found_hidden.shape == (15, 4)
    assert found_meta["latent_steps"] == 10


def test_incremental_latent_execution(tmp_path, mock_model):
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("42", True, None)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/Qwen3-8B",
        method="latent_mas",
        prompt="sequential",
        seed=42,
        temperature=0.6,
        top_p=0.95,
        max_new_tokens=256,
        single_intervention=True,
        latent_steps=20,
        latent_step_checkpoint_interval=10,
        no_progress=True,
    )
    runner = SingleInterventionRunner(mock_model, args, evaluator)
    checkpoints = {}

    def test_cb(step, kv, hidden, meta):
        checkpoints[step] = (kv, hidden, meta)

    item = {"question": "What is 2+2?", "gold": "4"}
    kv, hidden, meta = runner.compute_common_latent(
        item=item,
        checkpoint_callback=test_cb,
    )
    # Target was 20 with interval 10, so step 10 should be in checkpoints
    assert 10 in checkpoints
    assert meta["latent_steps"] == 20
    assert hidden.shape[0] == 3 + 20

    # Now simulate incremental extension from step 20 to 30
    args.latent_steps = 30
    checkpoints_ext = {}

    def test_cb_ext(step, kv, hidden, meta):
        checkpoints_ext[step] = (kv, hidden, meta)

    init_kv, init_hidden, init_meta = kv, hidden, meta
    mock_model.model.reset_mock()
    kv30, hidden30, meta30 = runner.compute_common_latent(
        item=dict(item, **init_meta),
        initial_kv=init_kv,
        initial_hiddens=init_hidden,
        initial_step=20,
        checkpoint_callback=test_cb_ext,
    )
    # Forward pass should only have run 10 times for the remaining steps (21..30)
    # Not prompt forward + 30 steps!
    assert mock_model.model.call_count == 10
    assert kv30 is not None
    assert meta30["latent_steps"] == 30
    assert hidden30.shape[0] == 3 + 30
