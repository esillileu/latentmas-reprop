# Configuration Guide

Configuration files provide reproducible presets for benchmarking experiments. They reside in the [`configs/`](../configs/) directory.

---

## 1. Naming Convention

Presets reside under use-case subdirectories in [`configs/`](../configs/) and follow a task-first naming pattern:
```
<use_case>/{task}_{model}[_{method}].yaml
```

- **`use_case`**:
  - `benchmark`
  - `intervention_kv`
  - `intervention_continuation`
  - `representation_acquisition`
- **`task`**:
  - Full dataset name (e.g. `gsm8k`, `secret_digit`, `aime2024`, `medqa`, `arc`, `mbpp_plus`, `humaneval_plus`)
- **`model`**:
  - `q30.6`: `Qwen/Qwen3-0.6B`
  - `q34`: `Qwen/Qwen3-4B`
  - `q38`: `Qwen/Qwen3-8B`
  - `q314`: `Qwen/Qwen3-14B`
  - `q3_sweep`: Multi-model sweep (`Qwen3-0.6B`, `4B`, `8B`, etc.)
- **`method`** *(included for `benchmark`; omitted when fixed to `latent_mas` in intervention/representation use cases)*:
  - `lm`: LatentMAS (Latent Multi-Agent System)
  - `tm`: TextMAS (Text Multi-Agent System)
  - `bs`: Baseline (Single Agent)

**Examples**:
- `benchmark/gsm8k_q38_lm.yaml`
- `benchmark/gsm8k_q38_bs.yaml`
- `intervention_kv/gsm8k_q38.yaml`
- `intervention_continuation/gsm8k_q38.yaml`
- `representation_acquisition/secret_digit_q3_sweep.yaml`

---

## 2. Configuration Schema & Parameters

A YAML configuration can define any argument accepted by `src/run/cli.py`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | *required* | `latent_mas`, `text_mas`, or `baseline` |
| `model_name` | `str` | *required* | Hugging Face model ID or local path |
| `task` | `str` | *required* | Benchmark dataset name |
| `prompt` | `str` | `"sequential"` | Agent topology: `sequential`, `hierarchical`, etc. |
| `max_samples` | `int` | `-1` | Maximum evaluation samples (`-1` for all) |
| `generate_bs` | `int` | `1` | Batch size during generation |
| `max_new_tokens` | `int` | `2048` | Maximum tokens generated per agent step |
| `temperature` | `float` | `0.0` | Sampling temperature (`0.0` for greedy decoding) |
| `top_p` | `float` | `1.0` | Nucleus sampling threshold |
| `latent_steps` | `int` | `4` | Number of latent reasoning steps per agent (LatentMAS only) |
| `latent_space_realign` | `bool` | `false` | Enable latent-to-embedding realignment projection |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `use_vllm` | `bool` | `false` | Use vLLM backend for faster generation |

Representation acquisition additionally accepts `representation_acquisition`, `context_modes`,
`acquisition_conditions`, `acquisition_cross_policy`, `save_hidden_states`, and
`save_raw_cache`. The canonical preset is `representation_acquisition/secret_digit_q3_sweep.yaml`; it requires
`task: secret_digit`, the transformers backend, and a positive `latent_steps` value.

The canonical secret-digit carrier modes are `full`, `prompt_only`, and
position-preserving `latent_only`. `drop_position_matched` isolates absolute-position
effects without retaining KV content. The same preset can enable the template-disjoint
sender-state probe with `probe_sender_latents`, `probe_prompt_templates`,
`probe_folds`, `probe_permutations`, `probe_backend`, `probe_batch_size`, and
`save_latent_states`. The canonical GPU backend is `torch`; it fits independent
L2-regularized multinomial classifiers in permutation batches. The default batch size
is 256, which is conservative for 32 GB VRAM and can be overridden for a particular
GPU. CUDA OOM automatically retries with half the batch size.

---

## 3. Example Presets

### LatentMAS (`configs/benchmark/gsm8k_q38_lm.yaml`)
```yaml
method: latent_mas
model_name: Qwen/Qwen3-8B
task: gsm8k
prompt: sequential
max_samples: 5
generate_bs: 4
latent_steps: 10
max_new_tokens: 256
temperature: 0.0
seed: 42
```

### Baseline (`configs/benchmark/gsm8k_q38_bs.yaml`)
```yaml
method: baseline
model_name: Qwen/Qwen3-8B
task: gsm8k
prompt: sequential
max_samples: -1
generate_bs: 2
max_new_tokens: 2048
temperature: 0.6
top_p: 0.95
seed: 42
```

### TextMAS (`configs/benchmark/gsm8k_q38_tm.yaml`)
```yaml
method: text_mas
model_name: Qwen/Qwen3-8B
task: gsm8k
prompt: sequential
max_samples: -1
generate_bs: 1
max_new_tokens: 2048
temperature: 0.6
top_p: 0.95
seed: 42
```

---

## 4. Creating a New Configuration

To create a new preset:
1. Create a file inside `configs/<use_case>/` following the naming convention, e.g. `configs/benchmark/aime2024_q38_lm.yaml`.
2. Define the desired hyperparameters in YAML format.
3. Run directly:
   ```bash
   just run -c benchmark/aime2024_q38_lm
   ```

Model inference precision is fixed to FP16 across hardware and backends, including
secondary HuggingFace models. BF16 capability does not change the selected dtype.
Numerical analysis routines retain their explicit computation dtypes.
