# LatentMAS Reproduction Task Runner

# Default recipe: list all available commands
default:
    @just --list

# Run multi-agent benchmark with options or YAML config
run *args:
    uv run python -m src.run {{args}}

# Run latent communication intervention experiment
run-intervention *args:
    uv run python -m src.run --intervention {{args}}

# Run secret-digit receiver acquisition
run-acquisition *args:
    uv run python -m src.run --acquisition {{args}}

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
