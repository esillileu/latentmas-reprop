"""Verify every model loading path requests FP16 explicitly."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from latentmas_reprop.infrastructure.models import loader


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
