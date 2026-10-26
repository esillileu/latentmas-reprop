"""Deterministic naming and experiment relative path construction."""

from typing import Any


def clean_model_name(model_name: str) -> str:
    """Extract clean, filesystem-safe model name (e.g. 'Qwen3-0.6B')."""
    raw = str(model_name).strip()
    if "/" in raw:
        return raw.split("/")[-1]
    if "\\" in raw:
        return raw.split("\\")[-1]
    return raw


def format_setting(value: Any) -> str:
    """Format a config value deterministically."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def build_params_stem(
    args: Any,
    dtype_name: str = "",
    backend: str = "",
) -> str:
    """Build deterministic, human-readable parameters directory name."""
    ls = int(getattr(args, "latent_steps", 0))
    seed = int(getattr(args, "seed", 42))
    temp = format_setting(float(getattr(args, "temperature", 0.6)))
    top = format_setting(float(getattr(args, "top_p", 0.95)))
    mt = int(getattr(args, "max_new_tokens", 2048))

    parts = [f"ls{ls}_seed{seed}_temp{temp}_top{top}_mt{mt}"]

    split = str(getattr(args, "split", "")).strip()
    if split and split != "test":
        parts.append(f"split{split}")

    if dtype_name:
        clean_dtype = str(dtype_name).replace("torch.", "")
        parts.append(clean_dtype)

    clean_backend = backend or ("vllm" if getattr(args, "use_vllm", False) else "")
    if clean_backend and clean_backend != "transformers":
        parts.append(clean_backend)

    if getattr(args, "think", False):
        parts.append("think")
    if getattr(args, "latent_space_realign", False):
        parts.append("realign")

    cross_policy = getattr(args, "cross_policy", None)
    if cross_policy and getattr(args, "intervention", False):
        parts.append(f"cross_{cross_policy}")

    if getattr(args, "smoke", False):
        parts.append("smoke")

    upstream = getattr(args, "upstream_steps", None)
    if upstream and (
        getattr(args, "receiver_reasoning", False)
        or getattr(args, "receiver_compute_preflight", False)
    ):
        steps_str = "-".join(str(s) for s in sorted(upstream))
        parts.append(f"upstream{steps_str}")

    budgets = getattr(args, "receiver_budgets", None)
    if budgets and getattr(args, "receiver_compute_preflight", False):
        budgets_str = "-".join(str(b) for b in budgets)
        parts.append(f"r{budgets_str}")

    return "_".join(parts)


def build_experiment_relpath(
    args: Any,
    dtype_name: str = "",
    backend: str = "",
) -> str:
    """Build canonical 5-level relative experiment path:

    <task>/<model>/<method>/<prompt>/<params>
    e.g. gsm8k/Qwen3-0.6B/latent_mas/sequential/ls20_seed42_temp0.6_top0.95_mt2048
    """
    task = str(getattr(args, "task", "gsm8k")).strip()
    model = clean_model_name(getattr(args, "model_name", "model"))
    method = str(getattr(args, "method", "latent_mas")).strip()
    prompt = str(getattr(args, "prompt", "sequential")).strip()

    params = build_params_stem(args, dtype_name=dtype_name, backend=backend)
    return f"{task}/{model}/{method}/{prompt}/{params}"


def is_overwrite_requested(args: Any) -> bool:
    """Check if overwrite is explicitly requested, robust against MagicMock."""
    val = getattr(args, "overwrite", False)
    return bool(val) if type(val) in (bool, int) else False
