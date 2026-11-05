"""Calculation budgets and prompt groups for continuation comparisons."""

import copy
from typing import Any

from ...domain.services.prompts.sequential_latent import (
    build_agent_message_sequential_latent_mas,
)

COMPARISON_CONDITIONS = (
    "baseline",
    "latent_only",
    "intervention_only",
    "latent_intervention",
)
SEQUENTIAL_CONDITIONS = {"intervention_only", "latent_intervention"}
SEQUENTIAL_ROLES = ("critic", "refiner", "judger")
UNPROMPTED_CONDITIONS = {"qe", "rqe", "qle", "rqle", "baseline", "latent_only"}


def branch_step(condition: str, args: Any) -> int:
    """Shared latent boundary before a condition branches."""
    if condition in {"qe", "rqe", "rqrqe", "baseline", "intervention_only"}:
        return 0
    if condition == "latent_intervention":
        return int(getattr(args, "intervention_latent_steps", 20))
    return int(getattr(args, "latent_steps", 10))


def comparison_groups(args: Any, conditions: list[str]) -> list[Any]:
    """Keep each initial prompt in its own reusable state tree."""
    groups = []
    for role, allowed in (
        ("single_agent", {"baseline", "latent_only"}),
        ("planner", SEQUENTIAL_CONDITIONS),
    ):
        selected = [condition for condition in conditions if condition in allowed]
        if selected:
            group = copy.copy(args)
            group.initial_input = "rq"
            group.initial_role = role
            group.intervention_conditions = selected
            groups.append(group)
    return groups


def sequential_prompts(model: Any, item: dict, args: Any) -> list[dict]:
    """Record the exact full chat prefix appended at each role transition."""
    stages = []
    for role in SEQUENTIAL_ROLES:
        messages = build_agent_message_sequential_latent_mas(
            role, item["question"], method="latent_mas", args=args
        )
        prompts, ids, _, tokens = model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        stages.append(
            {
                "role": role,
                "prompt": prompts[0],
                "token_ids": ids[0].cpu().tolist(),
                "tokens": tokens[0],
            }
        )
    return stages
