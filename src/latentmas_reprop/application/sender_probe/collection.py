"""Collect detached sender states without training the language model."""

from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm

PROMPT_TEMPLATES = (
    "Remember the private digit {digit} for the next agent.",
    "The secret number assigned to you is {digit}. Preserve it.",
    "Your private value is {digit}. Keep this information for another agent.",
    "Hold the digit {digit} in memory so a later agent can recover it.",
    "Privately retain {digit}; it must be communicated to the next agent.",
    "Store the confidential digit {digit} for the receiver.",
    "The hidden value is {digit}. Maintain it for downstream use.",
    "Memorize {digit} as the private number for the following agent.",
    "Keep {digit} as your secret digit until it is passed onward.",
    "Encode the private number {digit} for another agent to read.",
    "Safeguard digit {digit} in your internal state for the receiver.",
    "Retain this confidential number: {digit}. Another agent needs it.",
    "Your assigned secret is {digit}; preserve its identity internally.",
    "Carry the private digit {digit} forward to the receiving agent.",
    "Internally remember that the secret number equals {digit}.",
    "Keep track of private value {digit} for later communication.",
    "The receiver must recover {digit}; hold that digit in memory.",
    "Record {digit} as the hidden number intended for the next agent.",
    "Maintain the confidential value {digit} in your latent state.",
    "Preserve digit {digit} as private information for the receiver.",
)


@dataclass(frozen=True)
class SenderStates:
    hidden_pre_realign: torch.Tensor
    latent_post_realign: torch.Tensor
    labels: torch.Tensor
    template_indices: torch.Tensor
    metadata: list[dict[str, Any]]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "hidden_pre_realign": self.hidden_pre_realign,
            "latent_post_realign": self.latent_post_realign,
            "labels": self.labels,
            "template_indices": self.template_indices,
            "metadata": self.metadata,
        }


def collect_sender_states(
    model: Any, *, latent_steps: int, template_count: int
) -> SenderStates:
    """Run frozen-model inference for every template/digit pair."""
    if not 5 <= template_count <= len(PROMPT_TEMPLATES):
        raise ValueError(
            f"probe_prompt_templates must be between 5 and {len(PROMPT_TEMPLATES)}"
        )
    pre: list[torch.Tensor] = []
    post: list[torch.Tensor] = []
    labels: list[int] = []
    template_indices: list[int] = []
    metadata: list[dict[str, Any]] = []
    inputs = [
        (template_index, template, digit)
        for template_index, template in enumerate(PROMPT_TEMPLATES[:template_count])
        for digit in range(10)
    ]
    with torch.inference_mode():
        for template_index, template, digit in tqdm(
            inputs, desc="Collecting sender latent states", unit="state"
        ):
            messages = [[{"role": "user", "content": template.format(digit=digit)}]]
            _, input_ids, attention_mask, _ = model.prepare_chat_batch(
                messages, add_generation_prompt=True
            )
            rollout = model.generate_latent_batch_with_states(
                input_ids, attention_mask=attention_mask, latent_steps=latent_steps
            )
            pre.append(rollout.hidden_pre_realign[0].detach().cpu())
            post.append(rollout.latent_post_realign[0].detach().cpu())
            labels.append(digit)
            template_indices.append(template_index)
            metadata.append(
                {
                    "state_index": len(metadata),
                    "template_index": template_index,
                    "template": template,
                    "digit": digit,
                }
            )
    return SenderStates(
        torch.stack(pre),
        torch.stack(post),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(template_indices, dtype=torch.long),
        metadata,
    )
