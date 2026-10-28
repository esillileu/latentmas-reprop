"""Prompt builders and variant registry for Intervention Continuation P1 (sender/initial prompt)."""

from functools import lru_cache
from typing import Any

import yaml

from ....infrastructure.paths.resolver import get_path_resolver
from .hierarchical_latent import build_agent_message_hierarchical_latent_mas
from .sequential_latent import build_agent_message_sequential_latent_mas

DEFAULT_SYSTEM_MESSAGE = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
)
DEFAULT_P1_VARIANT = "planner"

FALLBACK_P1_VARIANTS: dict[str, dict[str, Any]] = {
    "planner": {
        "description": "Default LatentMAS Sequential Planner prompt",
        "role_builder": "planner",
        "template": (
            "You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.\n\n"
            "Question: {question}\n\n"
            "Your outlined plan should be concise with a few bulletpoints for each step. Do not produce the final answer.\n"
            "Now output your plan to solve the question below:\n"
        ),
    },
    "direct_solve": {
        "description": "Single-agent style direct step-by-step reasoning and boxed final answer prompt",
        "template": (
            "Target Question: {question}\n\n"
            "You are a helpful assistant.\n\n"
            "You must reason step-by-step to solve the provided Target Question without outputting other irrelevant information.\n\n"
            "Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n"
        ),
    },
    "question_only": {
        "description": "Raw question only without role or formatting instructions",
        "template": "Question: {question}\n",
    },
    "latent_think": {
        "description": "Instruct the model to reason silently in latent steps before answering",
        "template": (
            "Question: {question}\n\n"
            "Think carefully step-by-step about how to solve this problem in your internal state before producing the final solution.\n"
        ),
    },
    "math_agent": {
        "description": "Hierarchical LatentMAS style math agent prompt",
        "template": (
            "You are a math agent. Given the input question, reason step-by-step and put the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.\n\n"
            "Input Question: {question}\n\n"
            "Your response:\n"
        ),
    },
}


@lru_cache(maxsize=4)
def _load_p1_variants_from_file(yaml_path_str: str) -> dict[str, dict[str, Any]]:
    variants = dict(FALLBACK_P1_VARIANTS)
    try:
        with open(yaml_path_str, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        file_variants = data.get("variants", {})
        if isinstance(file_variants, dict):
            for k, v in file_variants.items():
                if isinstance(v, dict):
                    variants[str(k)] = v
    except Exception:
        pass
    return variants


def load_p1_variants() -> dict[str, dict[str, Any]]:
    """Load available P1 prompt variants from configs/prompts/intervention_p1.yaml with fallback."""
    try:
        resolver = get_path_resolver()
        yaml_file = resolver.configs_dir / "prompts" / "intervention_p1.yaml"
        if yaml_file.is_file():
            return _load_p1_variants_from_file(str(yaml_file.resolve()))
    except Exception:
        pass
    return dict(FALLBACK_P1_VARIANTS)


def resolve_p1_variant_id(args: Any) -> str:
    """Resolve the active P1 variant identifier for metadata and path construction."""
    custom_p1 = getattr(args, "p1_prompt", None)
    if custom_p1:
        return "custom"
    p1_var = getattr(args, "p1_variant", None)
    if p1_var:
        return str(p1_var).strip()
    sender_role = getattr(args, "sender_role", None)
    if sender_role and str(sender_role).strip() != DEFAULT_P1_VARIANT:
        return str(sender_role).strip()
    return DEFAULT_P1_VARIANT


def build_intervention_p1_messages(
    question: str,
    args: Any,
) -> list[dict[str, str]]:
    """Build chat messages for the P1 (initial/sender) prompt in intervention continuation."""
    custom_p1 = getattr(args, "p1_prompt", None)
    if custom_p1:
        q_text = str(custom_p1).format(question=question)
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
            {"role": "user", "content": q_text},
        ]

    variant_name = resolve_p1_variant_id(args)
    prompt_style = getattr(args, "prompt", "sequential")

    # Standard role-based builders ("planner", "critic", "refiner", "judger")
    if variant_name in {"planner", "critic", "refiner", "judger"}:
        builder = (
            build_agent_message_hierarchical_latent_mas
            if prompt_style == "hierarchical"
            else build_agent_message_sequential_latent_mas
        )
        return builder(
            role=variant_name,
            question=question,
            method="latent_mas",
            args=args,
        )

    variants = load_p1_variants()
    if variant_name not in variants:
        available = ", ".join(sorted(variants.keys()))
        raise ValueError(
            f"Unknown P1 prompt variant '{variant_name}'. Available variants: {available}"
        )

    spec = variants[variant_name]
    sys_msg = str(spec.get("system_message", DEFAULT_SYSTEM_MESSAGE))
    tmpl = str(spec.get("template", "Question: {question}"))
    user_content = tmpl.format(question=question)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_content},
    ]
