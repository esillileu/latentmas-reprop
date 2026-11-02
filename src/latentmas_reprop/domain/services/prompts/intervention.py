"""Initial RQ and Q prompts and their condition sets."""

from functools import lru_cache
from typing import Any

import yaml

from ....infrastructure.paths.resolver import get_path_resolver
from .hierarchical_latent import build_agent_message_hierarchical_latent_mas
from .sequential_latent import build_agent_message_sequential_latent_mas


@lru_cache(maxsize=4)
def _load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or not isinstance(data.get("roles"), dict):
        raise ValueError("Initial roles must be a mapping")
    return data


def load_intervention_config() -> dict[str, Any]:
    path = get_path_resolver().configs_dir / "prompts" / "intervention.yaml"
    return _load_config(str(path.resolve()))


def initial_input(args: Any) -> str:
    value = str(getattr(args, "initial_input", "rq")).lower()
    if value not in {"rq", "q"}:
        raise ValueError("Initial input must be rq or q")
    return value


def initial_role(args: Any) -> str | None:
    if initial_input(args) == "q":
        return None
    role = str(getattr(args, "initial_role", "planner"))
    if role not in load_intervention_config()["roles"]:
        raise ValueError(f"Unknown initial role '{role}'")
    return role


def initial_conditions(args: Any, *, defaults: bool = False) -> list[str]:
    config = load_intervention_config()
    initial_role(args)
    conditions = config["condition_sets"][initial_input(args)]
    return list(conditions["default" if defaults else "allowed"])


def build_initial_messages(question: str, args: Any) -> list[dict[str, str]]:
    config = load_intervention_config()
    role = initial_role(args)
    spec = config["roles"][role] if role else {"template": config["question_template"]}
    builder_role = spec.get("role_builder")
    if builder_role:
        builder = (
            build_agent_message_hierarchical_latent_mas
            if getattr(args, "prompt", "sequential") == "hierarchical"
            else build_agent_message_sequential_latent_mas
        )
        return builder(
            role=builder_role, question=question, method="latent_mas", args=args
        )
    return [
        {
            "role": "system",
            "content": spec.get("system_message", config["default_system_message"]),
        },
        {"role": "user", "content": spec["template"].format(question=question)},
    ]
