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

# Run test suite
test *args:
    uv run pytest {{args}}

# Run ruff linter
lint *args:
    uv run ruff check {{args}}

# Run ruff code formatter
format *args:
    uv run ruff format {{args}}

# Clean execution cache
clean-cache:
    rm -rf .cache/
