"""Stable identity of one benchmark configuration."""

from typing import Any

from ...domain.models import compute_sample_key

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


def format_setting(value: Any) -> str:
    """Format a config value so equivalent numbers share one run id."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def build_run_id(args: Any, dtype_name: str) -> str:
    """Identify a run by method, model, task, sampling, and dtype.

    ``max_samples`` stays at the requested value, including ``-1``. A five-sample
    smoke run therefore cannot overwrite or resume a full-test run.
    """
    model = str(args.model_name).replace("/", "_").replace("\\", "_")
    return (
        f"{args.method}__{model}__{args.task}__{args.prompt}"
        f"__ls{int(getattr(args, 'latent_steps', 0))}__seed{int(args.seed)}__{dtype_name}"
        f"__mt{int(args.max_new_tokens)}"
        f"_temp{format_setting(float(args.temperature))}"
        f"_top{format_setting(float(args.top_p))}"
        f"_bs{int(args.generate_bs)}_ms{int(args.max_samples)}"
        f"_think{format_setting(bool(getattr(args, 'think', False)))}"
        f"_realign{format_setting(bool(getattr(args, 'latent_space_realign', False)))}"
        f"_vllm{format_setting(bool(getattr(args, 'use_vllm', False)))}"
        f"_split{args.split}"
    )


def sample_key_for(args: Any, item: dict) -> str:
    """Stable sample id shared by a fresh run and a resumed run."""
    return compute_sample_key(args.task, args.split, item["question"])


def config_snapshot(args: Any) -> dict[str, Any]:
    """JSON-safe copy of the settings that define a benchmark run."""
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {key: values.get(key) for key in _RUN_FIELDS if key in values}
