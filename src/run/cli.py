import argparse
from typing import Any

import yaml

from latentmas_reprop.infrastructure.paths import get_path_resolver


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
        help="Path or name of YAML config file in configs/ (e.g. -c latent_mas_gsm8k)",
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
        choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
        default=defaults.get("model_name"),
        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').",
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

    return parser


def parse_args(args=None) -> argparse.Namespace:
    """Parse configuration with YAML template loading and CLI overrides."""
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

    parser = build_parser(defaults=defaults)
    parsed = parser.parse_args(args)

    if parsed.method == "latent_mas" and parsed.use_vllm:
        parsed.use_second_HF_model = True
        parsed.enable_prefix_caching = True

    return parsed
