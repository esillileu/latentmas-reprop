"""Stable identity of one benchmark configuration."""

from typing import Any

from ...domain.models import compute_sample_key
from ...infrastructure.paths.naming import (
    build_experiment_relpath,
)

_RUN_FIELDS = (
    "method",
    "model_name",
    "task",
    "prompt",
    "latent_steps",
    "seed",
    "max_new_tokens",
    "temperature",
    "top_p",
    "generate_bs",
    "max_samples",
    "think",
    "latent_space_realign",
    "use_vllm",
    "split",
)


def build_run_id(args: Any, dtype_name: str = "") -> str:
    """Identify a run by its canonical relative experiment directory path."""
    return build_experiment_relpath(args, dtype_name=dtype_name)


def sample_key_for(args: Any, item: dict, idx: int | None = None) -> str:
    """Zero-padded sample index (e.g. '000000') or fallback key."""
    if idx is not None:
        return f"{int(idx):06d}"
    if isinstance(item, dict) and "sample_index" in item:
        return f"{int(item['sample_index']):06d}"
    return compute_sample_key(args.task, args.split, item.get("question", ""))


def config_snapshot(args: Any) -> dict[str, Any]:
    """JSON-safe copy of the settings that define a benchmark run."""
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {key: values.get(key) for key in _RUN_FIELDS if key in values}
