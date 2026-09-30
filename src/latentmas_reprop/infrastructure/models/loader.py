"""Backend loader utilities for HuggingFace and vLLM models."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dtype import resolve_model_dtype


def ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    """Ensure tokenizer has a pad token."""
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


def load_hf_causal_lm(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.dtype]:
    """Load HuggingFace causal LM and tokenizer.

    BF16 is used only when the selected CUDA device supports it. V100-class
    GPUs use FP16. CPU loads use FP32. The returned dtype is the one requested
    from ``from_pretrained``.
    """
    selected = dtype if dtype is not None else resolve_model_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ensure_pad_token(tokenizer)
    with torch.no_grad():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=selected,
        )
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    parameter_dtype = next(model.parameters()).dtype
    if parameter_dtype != selected:
        raise RuntimeError(
            f"Loaded {model_name} as {parameter_dtype}, expected {selected}."
        )
    return tokenizer, model, selected


def init_vllm_backend(
    model_name: str,
    args: Any,
) -> tuple[
    Any, AutoTokenizer, AutoModelForCausalLM | None, torch.nn.Module | None, str | None
]:
    """Initialize vLLM engine, tokenizer, and optional secondary HF model."""
    from vllm import LLM

    tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
    gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

    print(f"[vLLM] Using vLLM backend for model {model_name}")
    if (
        getattr(args, "enable_prefix_caching", False)
        and getattr(args, "method", "") == "latent_mas"
    ):
        vllm_engine = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
            enable_prefix_caching=True,
            enable_prompt_embeds=True,
        )
    else:
        vllm_engine = LLM(
            model=model_name,
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=gpu_util,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    ensure_pad_token(tokenizer)

    use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
    hf_model = None
    embedding_layer = None
    hf_device = None
    if use_second_hf:
        hf_device = getattr(args, "device2", "cuda:1")
        hf_dtype = resolve_model_dtype(torch.device(hf_device))
        hf_model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=hf_dtype,
            )
            .to(hf_device)
            .eval()
        )
        embedding_layer = hf_model.get_input_embeddings()
    elif getattr(args, "latent_space_realign", False):
        raise ValueError(
            "latent_space_realign requires --use_second_HF_model when using vLLM backend."
        )

    return vllm_engine, tokenizer, hf_model, embedding_layer, hf_device
