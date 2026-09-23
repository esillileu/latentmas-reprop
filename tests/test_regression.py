import argparse
from unittest.mock import MagicMock

import torch

from latentmas_reprop.domain.services.latent_mas import LatentMASMethod


def test_latent_mas_regression_path():
    """Verify that build_latent_contexts + decode_with_context produces the exact same execution as run_batch."""
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.prepare_chat_batch.return_value = (
        ["prompt1"],
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[1, 1, 1]]),
        [["token1", "token2", "token3"]],
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.convert_ids_to_tokens.return_value = ["t1", "t2", "t3"]
    mock_model.tokenizer = mock_tokenizer
    mock_model.generate_latent_batch.return_value = (torch.zeros(1, 1, 4, 8),)
    mock_model.generate_text_batch.return_value = (
        ["The final answer is \\boxed{42}."],
        None,
        [8],
    )

    args = argparse.Namespace(
        model_name="Qwen/Qwen3-0.6B",
        task="gsm8k",
        prompt="sequential",
        think=False,
        latent_only=False,
        sequential_info_only=False,
    )

    method = LatentMASMethod(
        mock_model,
        latent_steps=2,
        judger_max_new_tokens=64,
        temperature=0.0,
        top_p=1.0,
        generate_bs=1,
        args=args,
    )

    item = {"question": "What is 40 + 2?", "gold": "42"}

    # Run direct run_batch
    res_direct = method.run_batch([item])

    # Run split two-step execution
    past_kv, traces = method.build_latent_contexts([item])
    res_split = method.decode_with_context(
        [item], past_kv=past_kv, initial_traces=traces
    )

    assert len(res_direct) == 1
    assert len(res_split) == 1
    assert res_direct[0]["prediction"] == res_split[0]["prediction"] == "42"
    assert res_direct[0]["correct"] == res_split[0]["correct"] is True
    assert res_direct[0]["raw_prediction"] == res_split[0]["raw_prediction"]
