"""vLLM integration and latent embedding generation utilities."""

from typing import Any

import torch


def vllm_generate_text(
    vllm_engine: Any,
    prompts: list[str],
    sampling_params: Any = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[str]:
    """Generate completions from text prompts using vLLM."""
    if vllm_engine is None:
        raise RuntimeError("vLLM engine is not initialized.")
    if sampling_params is None:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
    outputs = vllm_engine.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def generate_with_latent_embeddings(
    vllm_engine: Any,
    tokenizer: Any,
    embedding_layer: torch.nn.Module,
    judger_prompts: list[str],
    past_embedding: torch.Tensor,
    sampling_params: Any,
    hf_device: torch.device | str,
    vllm_device: torch.device | str,
    model_name: str,
) -> list[str]:
    """Splice latent embeddings into prompt embeddings and generate via vLLM."""
    if vllm_engine is None:
        raise RuntimeError("vLLM engine is not initialized.")

    judger_encoded = tokenizer(
        judger_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )["input_ids"].to(hf_device)

    curr_prompt_emb = embedding_layer(judger_encoded).squeeze(0).to(vllm_device)

    assert "Qwen" in model_name or "qwen" in model_name, (
        "latent_embedding_position is only supported for Qwen models currently."
    )

    len_of_left = []
    for p in judger_prompts:
        idx = p.find("<|im_start|>user\n")
        left = p[: idx + len("<|im_start|>user\n")]
        len_of_left.append(len(tokenizer(left)["input_ids"]))

    B, _L, H = curr_prompt_emb.shape
    past_embedding = past_embedding.to(vllm_device)

    whole_prompt_emb_list = []
    for i in range(B):
        insert_idx = len_of_left[i]
        left_emb = curr_prompt_emb[i, :insert_idx, :]
        right_emb = curr_prompt_emb[i, insert_idx:, :]
        combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
        whole_prompt_emb_list.append(combined)

    max_len = max(x.shape[0] for x in whole_prompt_emb_list)
    whole_prompt_emb = torch.stack(
        [
            torch.cat(
                [x, torch.zeros(max_len - x.shape[0], H, device=x.device)],
                dim=0,
            )
            for x in whole_prompt_emb_list
        ]
    )

    prompt_embeds_list = [{"prompt_embeds": embeds} for embeds in whole_prompt_emb]
    outputs = vllm_engine.generate(prompt_embeds_list, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]
