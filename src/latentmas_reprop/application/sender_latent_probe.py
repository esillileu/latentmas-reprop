"""Template-disjoint linear probe for secret-digit sender latent states."""

import random
from dataclasses import dataclass
from typing import Any

import torch

from ..infrastructure.models.model_wrapper import ModelWrapper

PROBE_PROMPT_TEMPLATES = (
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
class SenderProbeResult:
    states: dict[str, torch.Tensor]
    metadata: list[dict[str, Any]]
    split: dict[str, Any]
    summary: dict[str, Any]
    confusion_matrices: dict[str, list[list[int]]]


def _fit_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    *,
    seed: int,
    epochs: int,
) -> tuple[float, list[list[int]]]:
    train_x = features[train_mask].float()
    test_x = features[test_mask].float()
    train_y = labels[train_mask]
    test_y = labels[test_mask]
    center = train_x.mean(dim=0, keepdim=True)
    scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - center) / scale
    test_x = (test_x - center) / scale
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        classifier = torch.nn.Linear(train_x.shape[1], 10)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.03)
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(classifier(train_x), train_y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = classifier(test_x).argmax(dim=-1)
    accuracy = float((predictions == test_y).float().mean().item())
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    for expected, predicted in zip(test_y.tolist(), predictions.tolist(), strict=True):
        confusion[expected, predicted] += 1
    return accuracy, confusion.tolist()


def run_sender_latent_probe(model: ModelWrapper, args: Any) -> SenderProbeResult:
    template_count = int(args.probe_prompt_templates)
    if not 2 <= template_count <= len(PROBE_PROMPT_TEMPLATES):
        raise ValueError(
            f"probe_prompt_templates must be between 2 and {len(PROBE_PROMPT_TEMPLATES)}"
        )
    templates = PROBE_PROMPT_TEMPLATES[:template_count]
    indices = list(range(template_count))
    random.Random(args.seed).shuffle(indices)
    train_count = int(template_count * float(args.probe_train_template_fraction))
    if not 1 <= train_count < template_count:
        raise ValueError("probe template split must leave train and test templates")
    train_templates = set(indices[:train_count])
    test_templates = set(indices[train_count:])

    pre_states: list[torch.Tensor] = []
    post_states: list[torch.Tensor] = []
    labels: list[int] = []
    metadata: list[dict[str, Any]] = []
    for template_index, template in enumerate(templates):
        for digit in range(10):
            messages = [[{"role": "user", "content": template.format(digit=digit)}]]
            _, input_ids, attention_mask, _ = model.prepare_chat_batch(
                messages, add_generation_prompt=True
            )
            rollout = model.generate_latent_batch_with_states(
                input_ids,
                attention_mask=attention_mask,
                latent_steps=args.latent_steps,
            )
            pre_states.append(rollout.hidden_pre_realign[0].detach().cpu())
            post_states.append(rollout.latent_post_realign[0].detach().cpu())
            labels.append(digit)
            metadata.append(
                {
                    "state_index": len(metadata),
                    "template_index": template_index,
                    "template": template,
                    "digit": digit,
                    "split": "train" if template_index in train_templates else "test",
                }
            )

    states = {
        "hidden_pre_realign": torch.stack(pre_states),
        "latent_post_realign": torch.stack(post_states),
    }
    label_tensor = torch.tensor(labels, dtype=torch.long)
    template_tensor = torch.tensor(
        [item["template_index"] for item in metadata], dtype=torch.long
    )
    train_mask = torch.tensor(
        [int(index) in train_templates for index in template_tensor], dtype=torch.bool
    )
    test_mask = torch.tensor(
        [int(index) in test_templates for index in template_tensor], dtype=torch.bool
    )
    metrics: dict[str, float] = {}
    confusion_matrices: dict[str, list[list[int]]] = {}
    for representation, tensor in states.items():
        metric_name = (
            "pre_realign" if representation == "hidden_pre_realign" else "post_realign"
        )
        for step in range(args.latent_steps):
            accuracy, confusion = _fit_linear_probe(
                tensor[:, step, :],
                label_tensor,
                train_mask,
                test_mask,
                seed=args.seed + step,
                epochs=int(args.probe_epochs),
            )
            key = f"probe/{metric_name}/step_{step + 1}/accuracy"
            metrics[key] = accuracy
            confusion_matrices[f"{metric_name}_step_{step + 1}"] = confusion
    split = {
        "seed": args.seed,
        "train_template_indices": sorted(train_templates),
        "test_template_indices": sorted(test_templates),
        "train_templates": [templates[index] for index in sorted(train_templates)],
        "test_templates": [templates[index] for index in sorted(test_templates)],
        "train_template_fraction": args.probe_train_template_fraction,
        "chance_accuracy": 0.1,
    }
    summary = {
        "metrics": metrics,
        "chance_accuracy": 0.1,
        "n_examples": len(metadata),
        "n_train_examples": int(train_mask.sum()),
        "n_test_examples": int(test_mask.sum()),
        "latent_steps": args.latent_steps,
        "representations": list(states),
    }
    return SenderProbeResult(
        states=states,
        metadata=metadata,
        split=split,
        summary=summary,
        confusion_matrices=confusion_matrices,
    )
