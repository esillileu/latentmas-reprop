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


def resolve_experiment_dtype_name(args: Any) -> str:
    """Resolve model precision string deterministically before model loading."""
    if hasattr(args, "dtype") and getattr(args, "dtype", None):
        return str(args.dtype).replace("torch.", "")
    if hasattr(args, "model_dtype") and getattr(args, "model_dtype", None):
        return str(args.model_dtype).replace("torch.", "")
    # Default model precision across this framework is float16
    return "float16"


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

    resolved_dtype = dtype_name or resolve_experiment_dtype_name(args)
    if resolved_dtype:
        clean_dtype = str(resolved_dtype).replace("torch.", "")
        parts.append(clean_dtype)

    clean_backend = backend or ("vllm" if getattr(args, "use_vllm", False) else "")
    if clean_backend and clean_backend != "transformers":
        parts.append(clean_backend)

    if getattr(args, "think", False):
        parts.append("think")
    if getattr(args, "latent_space_realign", False):
        parts.append("realign")

    # Differentiating experiment types and calculation options
    if getattr(args, "intervention_continuation", False):
        parts.append("intervention_continuation")
        p1_custom = getattr(args, "p1_prompt", None)
        p1_var = getattr(args, "p1_variant", None) or getattr(args, "sender_role", None)
        if p1_custom:
            clean_p1 = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in str(p1_custom).strip()
            )
            clean_p1 = "_".join(filter(None, clean_p1.split("_")))[:20]
            if clean_p1:
                parts.append(f"p1_{clean_p1}")
        elif p1_var and str(p1_var).strip() != "planner":
            clean_var = "".join(
                c if c.isalnum() or c in ("-", "_") else "_"
                for c in str(p1_var).strip()
            )
            clean_var = "_".join(filter(None, clean_var.split("_")))
            if clean_var:
                parts.append(f"p1_{clean_var}")
        sig = getattr(args, "control_signal", None)
        if sig and sig != "\nSolution:":
            clean_sig = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in sig.strip())
            clean_sig = "_".join(filter(None, clean_sig.split("_")))
            if clean_sig:
                parts.append(f"sig_{clean_sig}")
        role = getattr(args, "receiver_role", None)
        if role and role != "judger":
            parts.append(f"role_{role}")
        plpe_p = getattr(args, "plpe_prompt", None)
        if plpe_p:
            clean_p = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(plpe_p).strip())
            clean_p = "_".join(filter(None, clean_p.split("_")))[:20]
            if clean_p:
                parts.append(f"plpe_{clean_p}")
    elif getattr(args, "intervention_kv", False):
        parts.append("intervention_kv")
        cross_policy = getattr(args, "cross_policy", None)
        if cross_policy:
            parts.append(f"cross_{cross_policy}")

    if getattr(args, "representation_acquisition", False):
        parts.append("representation_acquisition")
        modes = getattr(args, "context_modes", None)
        if modes:
            modes_str = "-".join(sorted(str(m) for m in modes))
            parts.append(f"modes_{modes_str}")
        conds = getattr(args, "acquisition_conditions", None)
        if conds:
            conds_str = "-".join(sorted(str(c) for c in conds))
            parts.append(f"cond_{conds_str}")

    if getattr(args, "representation_trajectory", False):
        parts.append("representation_trajectory")

    if getattr(args, "smoke", False):
        parts.append("smoke")

    return "_".join(parts)


def build_experiment_relpath(
    args: Any,
    dtype_name: str = "",
    backend: str = "",
) -> str:
    """Build canonical 5-level relative experiment path:

    <task>/<model>/<method>/<prompt>/<params>
    e.g. gsm8k/Qwen3-0.6B/latent_mas/sequential/ls20_seed42_temp0.6_top0.95_mt2048_float16
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
