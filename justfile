# LatentMAS Reproduction Task Runner

# Default recipe: list all available commands
default:
    @just --list

# Run multi-agent benchmark with options or YAML config
run *args:
    uv run python -m src.run {{args}}

# Run reproduction benchmark suite in model size order (0.6B -> 4B -> 8B -> 14B)
lmas action="reprop" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{action}}" = "reprop" ]; then
        echo "=== [1/4] Running Reproduction: Qwen3-0.6B ==="
        uv run python -m src.run -c lmas/reprop/bs_q30.6_gsm8k {{args}}
        uv run python -m src.run -c lmas/reprop/lm_q30.6_gsm8k {{args}}
        echo "=== [2/4] Running Reproduction: Qwen3-4B ==="
        uv run python -m src.run -c lmas/reprop/bs_q34_gsm8k {{args}}
        uv run python -m src.run -c lmas/reprop/lm_q34_gsm8k {{args}}
        echo "=== [3/4] Running Reproduction: Qwen3-8B ==="
        uv run python -m src.run -c lmas/reprop/bs_q38_gsm8k {{args}}
        uv run python -m src.run -c lmas/reprop/lm_q38_gsm8k {{args}}
        echo "=== [4/4] Running Reproduction: Qwen3-14B ==="
        uv run python -m src.run -c lmas/reprop/bs_q314_gsm8k {{args}}
        uv run python -m src.run -c lmas/reprop/lm_q314_gsm8k {{args}}
    else
        echo "Unknown action: {{action}}. Usage: just lmas reprop [args...]"
        exit 1
    fi

# Run latent communication intervention experiment
run-intervention *args:
    uv run python -m src.run --intervention {{args}}

# Run secret-digit receiver acquisition
run-acquisition *args:
    uv run python -m src.run --acquisition {{args}}

# Analyze previously collected sender latent states
analyze-sender-probe *args:
    uv run python -m src.run.sender_probe {{args}}

# Run test suite
test *args:
    uv run pytest {{args}}

# Run linters (ruff + flck)
lint *args:
    uv run ruff check {{args}}
    flck

# Run ruff code formatter
format *args:
    uv run ruff format {{args}}

# Clean execution cache
clean-cache:
    rm -rf .cache/

# Launch local MLflow UI pointing to SQLite backend
mlflow-ui *args:
    uv run mlflow ui --backend-store-uri sqlite:///.cache/mlflow.db --default-artifact-root .cache/mlflow_artifacts {{args}}
