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
# Run LatentMAS on GSM8K using Qwen3-8B
just run -c lmas/reprop/lm_q38_gsm8k

# Run the TextMAS and baseline matrix on GSM8K (Qwen3-8B)
just run -c lmas/reprop/bs_q38_gsm8k

# Run Single-Agent Baseline on GSM8K
just run -c bs_q30.6_gsm8k
```

> [!TIP]
> You do not need to append `.yaml` or provide the full path. `-c lmas/reprop/lm_q38_gsm8k`
> resolves against `configs/lmas/reprop/lm_q38_gsm8k.yaml`, and nested presets such as
> `-c lmas/secret_digit/preflight` resolve against
> `configs/lmas/secret_digit/preflight.yaml`. Full paths or arbitrary YAML files can
> also be passed.

---

## 3. Overriding Configuration Options

### Receiver acquisition

The synthetic secret-digit experiment uses the transformers KV cache and next-token
logits directly:

```bash
just run-acquisition -c lmas/secret_digit/preflight \
  --model_name Qwen/Qwen3-0.6B --latent_steps 4
```

Its default matrix decomposes `full`, `prompt_only`, and position-preserving
`latent_only` carriers under `own,cross`, plus `drop` and `drop_position_matched`.
Override subsets with `--carrier_modes` and `--acquisition_conditions`.
Hidden states and raw sender caches are only persisted when explicitly enabled.
The preset saves detached sender states and uses grouped five-fold logistic regression
with 5,000 within-template permutations. Reanalyze states without model inference
with:

```bash
just analyze-sender-probe --states PATH --source-config CONFIG
MLFLOW_TRACKING_URI=URI just analyze-sender-probe \
  --source-run-id RUN_ID --source-artifact-path probe/sender_latent_states.pt
```

To replace a legacy run containing the invalid single-split probe, use:

```bash
MLFLOW_TRACKING_URI=URI just analyze-sender-probe \
  --source-run-id RUN_ID \
  --replace-source-run \
  --backend torch \
  --permutations 5000 \
  --batch-size 256
```

The command copies all non-probe parameters and metric histories, acquisition
artifacts, original start/end timestamps, and trace links into a replacement run. It
then records canonical `probe/results.json` and `probe/null_statistics.json`, verifies
trace ownership, and only then soft-deletes the legacy run. Progress and ETA are
printed after every permutation batch. Larger batches are not necessarily faster for
joint batched L-BFGS, so 256 remains the default even on a 32 GB GPU.

Any option defined in a config file can be overridden directly from the command line:

```bash
# Override the number of evaluated samples:
just run -c lmas/reprop/lm_q38_gsm8k --max_samples 10

# Override temperature and generation batch size:
just run -c lmas/reprop/lm_q38_gsm8k --temperature 0.7 --generate_bs 2

# Enable latent space realignment on top of the preset:
just run -c lmas/reprop/lm_q38_gsm8k --latent_space_realign
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
| `just run-intervention <options>` | Execute the latent-cache intervention harness |
| `just run-acquisition <options>` | Execute secret-digit receiver acquisition |
| `just analyze-sender-probe <options>` | Analyze saved sender states without inference |
| `just test` | Run the complete pytest test suite |
| `just lint` | Run Ruff linter checks |
| `just format` | Format code using Ruff formatter |
| `just clean-cache` | Clear execution runs and model caches in `.cache/` |
