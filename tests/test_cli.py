try:
    from src.run.cli import parse_args
except ModuleNotFoundError:
    from run.cli import parse_args


def test_parse_args_latent_mas():
    args = parse_args([
        "--method", "latent_mas",
        "--model_name", "Qwen/Qwen3-0.6B",
        "--task", "gsm8k",
        "--prompt", "sequential",
        "--max_samples", "5",
        "--generate_bs", "1",
        "--latent_steps", "4",
        "--max_new_tokens", "256",
    ])
    assert args.method == "latent_mas"
    assert args.model_name == "Qwen/Qwen3-0.6B"
    assert args.task == "gsm8k"
    assert args.prompt == "sequential"
    assert args.max_samples == 5
    assert args.generate_bs == 1
    assert args.latent_steps == 4
    assert args.max_new_tokens == 256


def test_parse_args_baseline():
    args = parse_args([
        "--method", "baseline",
        "--model_name", "Qwen/Qwen3-0.6B",
        "--task", "gsm8k",
    ])
    assert args.method == "baseline"
    assert args.model_name == "Qwen/Qwen3-0.6B"
