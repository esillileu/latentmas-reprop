import pytest

try:
    from src.run.cli import parse_args
except ModuleNotFoundError:
    from run.cli import parse_args


def test_parse_args_latent_mas():
    args = parse_args(
        [
            "--method",
            "latent_mas",
            "--model_name",
            "Qwen/Qwen3-0.6B",
            "--task",
            "gsm8k",
            "--prompt",
            "sequential",
            "--max_samples",
            "5",
            "--generate_bs",
            "1",
            "--latent_steps",
            "4",
            "--max_new_tokens",
            "256",
        ]
    )
    assert args.method == "latent_mas"
    assert args.model_name == "Qwen/Qwen3-0.6B"
    assert args.task == "gsm8k"
    assert args.prompt == "sequential"
    assert args.max_samples == 5
    assert args.generate_bs == 1
    assert args.latent_steps == 4
    assert args.max_new_tokens == 256


def test_parse_args_baseline():
    args = parse_args(
        [
            "--method",
            "baseline",
            "--model_name",
            "Qwen/Qwen3-0.6B",
            "--task",
            "gsm8k",
        ]
    )
    assert args.method == "baseline"
    assert args.model_name == "Qwen/Qwen3-0.6B"


def test_parse_args_config_preset():
    # Load defaults from configs/latent_mas_gsm8k.yaml using --config
    args = parse_args(["--config", "latent_mas_gsm8k"])
    assert args.method == "latent_mas"
    assert args.model_name == "Qwen/Qwen3-0.6B"
    assert args.task == "gsm8k"
    assert args.prompt == "sequential"
    assert args.max_samples == 5
    assert args.latent_steps == 4


def test_parse_args_config_with_override():
    # Override max_samples and temperature using -c
    args = parse_args(
        [
            "-c",
            "latent_mas_gsm8k",
            "--max_samples",
            "20",
            "--temperature",
            "0.8",
        ]
    )
    assert args.method == "latent_mas"
    assert args.max_samples == 20
    assert args.temperature == 0.8
    assert args.latent_steps == 4


def test_parse_args_template_alias():
    # Verify --template alias still works
    args = parse_args(["--template", "latent_mas_gsm8k"])
    assert args.method == "latent_mas"


def test_parse_args_config_not_found():
    with pytest.raises(FileNotFoundError):
        parse_args(["--config", "non_existent_preset_file"])
