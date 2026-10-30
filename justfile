# LatentMAS Reproduction Task Runner

export PYTORCH_CUDA_ALLOC_CONF := "expandable_segments:True"

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
    if [ "{{action}}" = "reprop" ] || [ "{{action}}" = "benchmark" ]; then
        echo "=== [1/4] Running Benchmark: Qwen3-0.6B ==="
        uv run python -m src.run -c benchmark/gsm8k_q30.6_bs {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q30.6_tm {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q30.6_lm {{args}}
        echo "=== [2/4] Running Benchmark: Qwen3-4B ==="
        uv run python -m src.run -c benchmark/gsm8k_q34_bs {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q34_tm {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q34_lm {{args}}
        echo "=== [3/4] Running Benchmark: Qwen3-8B ==="
        uv run python -m src.run -c benchmark/gsm8k_q38_bs {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q38_tm {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q38_lm {{args}}
        echo "=== [4/4] Running Benchmark: Qwen3-14B ==="
        uv run python -m src.run -c benchmark/gsm8k_q314_bs {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q314_tm {{args}}
        uv run python -m src.run -c benchmark/gsm8k_q314_lm {{args}}
    else
        echo "Unknown action: {{action}}. Usage: just lmas [benchmark|reprop] [args...]"
        exit 1
    fi

# Run KV intervention experiment (own, cross, drop, zero)
run-intervention-kv *args:
    uv run python -m src.run --intervention_kv {{args}}

# Run intervention continuation analysis (PLE, PLCE, PLPE)
run-intervention-continuation *args:
    uv run python -m src.run --intervention_continuation {{args}}

# Inspect legacy intervention continuation data; pass --apply to publish
migrate-intervention-continuation *args:
    uv run python -m tools.migrations.intervention_continuation.migrate {{args}}

# Run secret-digit representation acquisition
run-representation-acquisition *args:
    uv run python -m src.run --representation_acquisition {{args}}

# Analyze previously collected sender latent states
analyze-representation-sender-probe *args:
    uv run python -m src.run.representation_sender_probe {{args}}

# Reanalyze saved acquisition observations from MLflow (no inference)
analyze-representation-acquisition *args:
    uv run python -m src.run.representation_acquisition_analysis {{args}}

# Presentation PNG/PDF figures from existing analysis CSVs (no inference)
plot-presentation *args:
    uv run --frozen python -m src.run.presentation_plots {{args}}

# BF16 latent-only KV trajectories; full execution requires a 32GB GPU
run-representation-trajectory *args:
    uv run --frozen python -m src.run.representation_trajectory {{args}}

# Read-only 60-point scatter from completed receiver trajectories
plot-representation-trajectory *args:
    uv run --frozen python -m src.run.representation_trajectory_plots {{args}}

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

# Clean volatile execution cache (safely preserving MLflow DB and persistent storage)
clean-cache:
    uv run python -c "from latentmas_reprop.infrastructure.cache.manager import get_cache_manager; get_cache_manager().clean_volatile_cache()"

# Launch local MLflow UI pointing to persistent SQLite backend
mlflow-ui *args:
    uv run mlflow ui --backend-store-uri sqlite:///storage/mlflow/mlflow.db --default-artifact-root storage/mlflow/mlflow_artifacts {{args}}

# Analyze saved secret-digit trajectories without inference
analyze-representation-trajectory *args:
    uv run --frozen python -m src.run.representation_trajectory_analysis {{args}}

# Strict offline 60-cell communication metrics, parity table and plots
analyze-stepwise-communication *args:
    uv run --frozen python -m src.run.stepwise_communication_analysis {{args}}

# One-time migration of results and MLflow references after the use case rename (dry run by default)
migrate-use-case-names *args:
    uv run python -m tools.migrations.use_case_names.migrate {{args}}
