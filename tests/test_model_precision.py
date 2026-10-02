"""Verify every model loading path requests FP16 explicitly."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from latentmas_reprop.infrastructure.models import loader


@pytest.mark.parametrize(
    "requested,expected", [(None, torch.float16), (torch.bfloat16, torch.bfloat16)]
)
def test_hf_explicit_bf16_preserves_fp16_default(monkeypatch, requested, expected):
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.__len__.return_value = 10
    model = Mock()
    model.get_input_embeddings.return_value.weight = torch.zeros(10, 2)
    model.parameters.return_value = iter([torch.zeros(1, dtype=expected)])
    factory = Mock(return_value=model)
    monkeypatch.setattr(
        loader.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", factory)
    _, _, actual = loader.load_hf_causal_lm(
        "model", torch.device("cpu"), dtype=requested
    )
    factory.assert_called_once_with("model", dtype=expected)
    assert actual == expected


@pytest.mark.parametrize("prefix_cache", [False, True])
@pytest.mark.parametrize("secondary", [False, True])
def test_vllm_and_secondary_hf_use_fp16(monkeypatch, prefix_cache, secondary):
    engine_factory = Mock()
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=engine_factory))
    tokenizer = SimpleNamespace(pad_token_id=0)
    monkeypatch.setattr(
        loader.AutoTokenizer, "from_pretrained", Mock(return_value=tokenizer)
    )
    hf_model = Mock()
    hf_model.to.return_value = hf_model
    hf_model.eval.return_value = hf_model
    hf_factory = Mock(return_value=hf_model)
    monkeypatch.setattr(loader.AutoModelForCausalLM, "from_pretrained", hf_factory)
    args = SimpleNamespace(
        method="latent_mas",
        enable_prefix_caching=prefix_cache,
        use_second_HF_model=secondary,
        device2="cpu",
    )
    loader.init_vllm_backend("model", args)
    assert engine_factory.call_args.kwargs["dtype"] == "float16"
    if secondary:
        hf_factory.assert_called_once_with("model", dtype=torch.float16)
    else:
        hf_factory.assert_not_called()
