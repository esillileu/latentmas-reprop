"""vLLM execution engine for LatentMAS method."""

from typing import Any

import torch


def run_latent_mas_vllm_batch(method: Any, items: list[dict]) -> list[dict]:
    """Execute a batch of items through vLLM for LatentMAS."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=method.temperature,
        top_p=method.top_p,
        max_tokens=method.judger_max_new_tokens,
    )
    _, embedding_record, agent_traces = method._forward_non_judger(
        items, record_embeddings=True
    )
    judger_agent = next((a for a in method.agents if a.role == "judger"), None)
    if judger_agent is None:
        raise RuntimeError("No judger agent found in self.agents")

    batch_messages = method._build_messages(judger_agent.role, items)
    prompts, _, _, _ = method.model.prepare_chat_batch(
        batch_messages, add_generation_prompt=True
    )
    past_embedding = torch.cat(embedding_record, dim=1).to(method.vllm_device)
    judger_prompts = (
        [f"{p}<think>" for p in prompts]
        if getattr(method.args, "think", False)
        else prompts
    )
    generated_texts = method.model.generate_with_latent_embeddings(
        judger_prompts,
        past_embedding,
        sampling_params,
        method.vllm_device,
    )
    for idx in range(len(items)):
        agent_traces[idx].append(
            {
                "name": judger_agent.name,
                "role": judger_agent.role,
                "input": judger_prompts[idx],
                "output": generated_texts[idx].strip(),
            }
        )
    return method._evaluate_results(items, generated_texts, agent_traces)
