import pytest

try:
    from src.run.cli import parse_args, parse_run_matrix
except ModuleNotFoundError:
    from run.cli import parse_args, parse_run_matrix


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "benchmark.yaml"
    path.write_text(
        "\n".join(
            (
                "method: latent_mas",
                "model_name: Qwen/Qwen3-0.6B",
                "task: gsm8k",
                "prompt: sequential",
                "max_samples: 5",
                "latent_steps: 4",
            )
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture
def matrix_config_path(tmp_path):
    path = tmp_path / "matrix.yaml"
    path.write_text(
        "\n".join(
            (
                "method:",
                "  - text_mas",
                "  - latent_mas",
                "model_name: Qwen/Qwen3-0.6B",
                "task: gsm8k",
                "latent_steps:",
                "  - 1",
                "  - 4",
                "  - 20",
            )
        ),
        encoding="utf-8",
    )
    return str(path)


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


def test_parse_args_config_preset(config_path):
    args = parse_args(["--config", config_path])
    assert args.method == "latent_mas"
    assert args.model_name == "Qwen/Qwen3-0.6B"
    assert args.task == "gsm8k"
    assert args.prompt == "sequential"
    assert args.max_samples == 5
    assert args.latent_steps == 4


def test_parse_args_config_with_override(config_path):
    # Override max_samples and temperature using -c
    args = parse_args(
        [
            "-c",
            config_path,
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


def test_parse_args_template_alias(config_path):
    # Verify --template alias still works
    args = parse_args(["--template", config_path])
    assert args.method == "latent_mas"


def test_parse_args_config_not_found():
    with pytest.raises(FileNotFoundError):
        parse_args(["--config", "non_existent_preset_file"])


def test_run_matrix_expands_list_fields_as_cartesian_product(matrix_config_path):
    runs = parse_run_matrix(["--config", matrix_config_path])

    assert [(run.method, run.latent_steps) for run in runs] == [
        ("text_mas", 1),
        ("text_mas", 4),
        ("text_mas", 20),
        ("latent_mas", 1),
        ("latent_mas", 4),
        ("latent_mas", 20),
    ]


def test_cli_override_collapses_sweep_dimension(matrix_config_path):
    runs = parse_run_matrix(["--config", matrix_config_path, "--method", "latent_mas"])

    assert [(run.method, run.latent_steps) for run in runs] == [
        ("latent_mas", 1),
        ("latent_mas", 4),
        ("latent_mas", 20),
    ]


def test_parse_args_rejects_multiple_runs(matrix_config_path):
    with pytest.raises(ValueError, match="expands to 6 runs"):
        parse_args(["--config", matrix_config_path])
