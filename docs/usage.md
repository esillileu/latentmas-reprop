# Usage & Execution Guide

This guide explains how to run benchmarks, use configuration presets, override settings, and utilize the `just` command runner.

---

## 1. Prerequisites

Ensure dependencies are installed via `uv`:
```bash
uv sync
```

Install `just` if not already installed:
```bash
# macOS
brew install just

# Linux / Pre-built binary
curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin
```

---

## 2. Running Benchmarks with `just`

The primary interface for running benchmarks is `just run`:

### 2.1 Using Preset Configurations (`-c`)
Configurations are stored in [`configs/`](../configs/) using the format:
```
{method}_{model}_{task}.yaml
```
- **Methods**: `lm` (LatentMAS), `tm` (TextMAS), `bs` (Baseline)
- **Model**: Shortened model name (e.g., `q30.6` for Qwen3-0.6B)
- **Task**: Full task name (e.g., `gsm8k`, `aime2024`, `medqa`)

Examples:
```bash
# Run LatentMAS on GSM8K using Qwen3-0.6B
just run -c lm_q30.6_gsm8k

# Run TextMAS on GSM8K
just run -c tm_q30.6_gsm8k

# Run Single-Agent Baseline on GSM8K
just run -c bs_q30.6_gsm8k
```

> [!TIP]
> You do not need to append `.yaml` or provide the full path. `-c lm_q30.6_gsm8k` will automatically resolve against `configs/lm_q30.6_gsm8k.yaml`. Full paths or arbitrary YAML files can also be passed.

---

## 3. Overriding Configuration Options

Any option defined in a config file can be overridden directly from the command line:

```bash
# Override the number of evaluated samples:
just run -c lm_q30.6_gsm8k --max_samples 10

# Override temperature and generation batch size:
just run -c lm_q30.6_gsm8k --temperature 0.7 --generate_bs 2

# Enable latent space realignment on top of the preset:
just run -c lm_q30.6_gsm8k --latent_space_realign
```

---

## 4. Running Without a Configuration File

If no `-c` / `--config` is specified, all required options must be passed as CLI flags:

```bash
just run \
  --method latent_mas \
  --model_name Qwen/Qwen3-0.6B \
  --task gsm8k \
  --prompt sequential \
  --max_samples 5 \
  --latent_steps 4 \
  --max_new_tokens 256
```

---

## 5. Available `just` Tasks

List all available tasks:
```bash
just --list
```

| Command | Description |
|---|---|
| `just run <options>` | Execute benchmark runner (`uv run python -m src.run <args>`) |
| `just test` | Run the complete pytest test suite |
| `just lint` | Run Ruff linter checks |
| `just format` | Format code using Ruff formatter |
| `just clean-cache` | Clear execution runs and model caches in `.cache/` |
