import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build and return command-line argument parser for benchmark runs."""
    parser = argparse.ArgumentParser(
        description="LatentMAS / TextMAS / Baseline Multi-Agent Benchmark Runner"
    )

    # Core args for experiments
    parser.add_argument(
        "--method",
        choices=["baseline", "text_mas", "latent_mas"],
        required=True,
        help="Which multi-agent method to run: 'baseline', 'text_mas', or 'latent_mas'.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B"],
        help="Model choices to use for experiments (e.g. 'Qwen/Qwen3-14B').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
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
        default="gsm8k",
        help="Dataset/task to evaluate. Controls which loader is used.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        choices=["sequential", "hierarchical"],
        default="sequential",
        help="Multi-agent system architecture: 'sequential' or 'hierarchical'.",
    )

    # Execution configuration
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument(
        "--latent_steps",
        type=int,
        default=0,
        help="Number of latent steps for LatentMAS method",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--generate_bs", type=int, default=20, help="Batch size for generation"
    )
    parser.add_argument(
        "--text_mas_context_length",
        type=int,
        default=-1,
        help="TextMAS context length limit",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Manually add think token in the prompt for LatentMAS",
    )
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument(
        "--use_vllm", action="store_true", help="Use vLLM backend for generation"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching in vLLM for latent_mas",
    )
    parser.add_argument(
        "--use_second_HF_model",
        action="store_true",
        help="Use a second HF model for latent generation in latent_mas",
    )
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="How many GPUs vLLM should shard the model across",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Target GPU memory utilization for vLLM",
    )

    return parser


def parse_args(args=None) -> argparse.Namespace:
    """Parse and post-process arguments."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.method == "latent_mas" and parsed.use_vllm:
        parsed.use_second_HF_model = True
        parsed.enable_prefix_caching = True

    return parsed
