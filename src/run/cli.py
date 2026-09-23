"""Command-line argument parser for benchmark runs."""

import argparse
import itertools
from typing import Any

import yaml

from latentmas_reprop.infrastructure.paths import get_path_resolver

from .cli_intervention import (
    add_acquisition_args,
    add_intervention_args,
    validate_intervention_and_acquisition_args,
)


def build_parser(defaults: dict[str, Any] | None = None) -> argparse.ArgumentParser:
    """Build and return command-line argument parser for benchmark runs."""
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="LatentMAS / TextMAS / Baseline Multi-Agent Benchmark Runner"
    )

    # Configuration / YAML preset
    parser.add_argument(
        "-c",
        "--config",
        "--template",
        dest="config_path",
        type=str,
        default=None,
        metavar="CONFIG",
        help="Path or name of a YAML config file in configs/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded run matrix without loading models or running experiments.",
    )

    # Core args for experiments
    method_required = "method" not in defaults
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas"],
        required=method_required,
        default=defaults.get("method"),
        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.",
    )
    model_name_required = "model_name" not in defaults
    parser.add_argument(
        "--model_name",
        type=str,
        required=model_name_required,
        default=defaults.get("model_name"),
        help="Model identifier to use for experiments (e.g. 'Qwen/Qwen3-0.6B', 'Qwen/Qwen2.5-7B-Instruct').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=defaults.get("max_samples", -1),
        help="Number of questions to evaluate; set -1 to use all samples.",
    )
    parser.add_argument(
        "--task",
        choices=[
            "gsm8k",
            "aime2024",
            "aime2025",
            "gpqa",
            "arc_easy",
            "arc_challenge",
            "mbppplus",
            "humanevalplus",
            "medqa",
            "secret_digit",
        ],
        default=defaults.get("task", "gsm8k"),
        help="Dataset/task to evaluate. Controls which loader is used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["sequential", "hierarchical"],
        default=defaults.get("prompt", "sequential"),
        help="Multi-agent system architecture: 'sequential' or 'hierarchical'.",
    )

    # Execution configuration
    parser.add_argument("--device", type=str, default=defaults.get("device", "cuda"))
    parser.add_argument("--split", type=str, default=defaults.get("split", "test"))
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=defaults.get("max_new_tokens", 4096),
    )
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=defaults.get("latent_steps", 0),
        help="Number of latent steps for LatentMAS method",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=defaults.get("temperature", 0.6),
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=defaults.get("top_p", 0.95),
    )
    parser.add_argument(
        "--generate_bs",
        type=int,
        default=defaults.get("generate_bs", 20),
        help="Batch size for generation",
    )
    parser.add_argument(
        "--text_mas_context_length",
        type=int,
        default=defaults.get("text_mas_context_length", -1),
        help="TextMAS context length limit",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        default=defaults.get("think", False),
        help="Manually add think token in the prompt for LatentMAS",
    )
    parser.add_argument(
        "--latent_space_realign",
        action="store_true",
        default=defaults.get("latent_space_realign", False),
    )
    parser.add_argument("--seed", type=int, default=defaults.get("seed", 42))

    # vLLM support
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        default=defaults.get("use_vllm", False),
        help="Use vLLM backend for generation",
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=defaults.get("enable_prefix_caching", False),
        help="Enable prefix caching in vLLM for latent_mas",
    )
    parser.add_argument(
        "--use_second_HF_model",
        action="store_true",
        default=defaults.get("use_second_HF_model", False),
        help="Use a second HF model for latent generation in latent_mas",
    )
    parser.add_argument(
        "--device2", type=str, default=defaults.get("device2", "cuda:1")
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=defaults.get("tensor_parallel_size", 1),
        help="How many GPUs vLLM should shard the model across",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=defaults.get("gpu_memory_utilization", 0.9),
        help="Target GPU memory utilization for vLLM",
    )

    # Modular options
    add_intervention_args(parser, defaults)
    add_acquisition_args(parser, defaults)

    return parser


def _load_defaults(args: list[str] | None) -> dict[str, Any]:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config", "-c", "--template", dest="config_path", type=str, default=None
    )
    pre_args, _ = pre_parser.parse_known_args(args)

    defaults: dict[str, Any] = {}
    if pre_args.config_path:
        resolver = get_path_resolver()
        config_file = resolver.resolve_config_path(pre_args.config_path)
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {pre_args.config_path} (resolved to {config_file})"
            )
        with open(config_file, encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                defaults = loaded
    return defaults


def _parse_args(args: list[str] | None, defaults: dict[str, Any]) -> argparse.Namespace:
    parser = build_parser(defaults=defaults)
    parsed = parser.parse_args(args)

    if parsed.method not in {"baseline", "text_mas", "latent_mas"}:
        parser.error(f"unsupported --method: {parsed.method}")

    if parsed.method == "latent_mas" and parsed.use_vllm:
        parsed.use_second_HF_model = True
        parsed.enable_prefix_caching = True

    validate_intervention_and_acquisition_args(parser, parsed)

    return parsed


def parse_run_matrix(args=None) -> list[argparse.Namespace]:
    """Expand top-level YAML lists into a Cartesian product of parsed runs."""
    argv = list(args) if args is not None else None
    defaults = _load_defaults(argv)
    dimensions = {
        key: value for key, value in defaults.items() if isinstance(value, list)
    }
    empty = [key for key, values in dimensions.items() if not values]
    if empty:
        raise ValueError(f"sweep dimensions cannot be empty: {', '.join(empty)}")
    if not dimensions:
        return [_parse_args(argv, defaults)]

    scalar_defaults = {
        key: value for key, value in defaults.items() if key not in dimensions
    }
    parsed_runs = []
    seen = set()
    for values in itertools.product(*dimensions.values()):
        run_defaults = scalar_defaults | dict(zip(dimensions, values, strict=True))
        parsed = _parse_args(argv, run_defaults)
        signature = tuple(
            sorted((key, repr(value)) for key, value in vars(parsed).items())
        )
        if signature not in seen:
            seen.add(signature)
            parsed_runs.append(parsed)
    return parsed_runs


def parse_args(args=None) -> argparse.Namespace:
    """Parse a configuration that resolves to exactly one run."""
    runs = parse_run_matrix(args)
    if len(runs) != 1:
        raise ValueError(
            f"configuration expands to {len(runs)} runs; use parse_run_matrix()"
        )
    return runs[0]
