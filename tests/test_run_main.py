"""Tests for matrix execution at the CLI entry point."""

from run import __main__ as run_main


def test_dry_run_prints_matrix_without_running_models(tmp_path, monkeypatch, capsys):
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "\n".join(
            (
                "method: latent_mas",
                "model_name:",
                "  - model-a",
                "  - model-b",
                "task: gsm8k",
                "latent_steps:",
                "  - 1",
                "  - 4",
                "  - 20",
            )
        ),
        encoding="utf-8",
    )
    executed = []
    monkeypatch.setattr(run_main, "run_benchmark", executed.append)

    run_main.main(["--config", str(config), "--dry-run"])

    output = capsys.readouterr().out
    assert "Run plan: 6 run(s)" in output
    assert "model_name=model-a, latent_steps=1" in output
    assert "model_name=model-b, latent_steps=20" in output
    assert not executed


def test_matrix_runs_sequentially(tmp_path, monkeypatch):
    config = tmp_path / "matrix.yaml"
    config.write_text(
        "\n".join(
            (
                "method:",
                "  - text_mas",
                "  - latent_mas",
                "model_name: model",
                "task: gsm8k",
                "latent_steps:",
                "  - 1",
                "  - 4",
            )
        ),
        encoding="utf-8",
    )
    executed = []
    monkeypatch.setattr(
        run_main,
        "run_benchmark",
        lambda args: executed.append((args.method, args.latent_steps)),
    )

    run_main.main(["--config", str(config)])

    assert executed == [
        ("text_mas", 1),
        ("text_mas", 4),
        ("latent_mas", 1),
        ("latent_mas", 4),
    ]
