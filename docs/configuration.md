# Configuration Guide

Configuration files provide reproducible presets for benchmarking experiments. They reside in the [`configs/`](../configs/) directory.

---

## 1. Naming Convention

Presets follow a concise naming pattern:
```
{method}_{model}_{task}.yaml
```

- **`method`**:
  - `lm`: LatentMAS (Latent Multi-Agent System)
  - `tm`: TextMAS (Text Multi-Agent System)
  - `bs`: Baseline (Single Agent)
- **`model`**:
  - `q30.6`: `Qwen/Qwen3-0.6B`
  - `q34`: `Qwen/Qwen3-4B`
  - `q38`: `Qwen/Qwen3-8B`
  - `q314`: `Qwen/Qwen3-14B`
- **`task`**:
  - Full dataset name (e.g. `gsm8k`, `aime2024`, `medqa`, `arc`, `mbpp_plus`, `humaneval_plus`)

**Examples**:
- `lmas/reprop/lm_q38_gsm8k.yaml`
- `lmas/reprop/bs_q38_gsm8k.yaml`
- `bs_q30.6_gsm8k.yaml`

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

Receiver acquisition additionally accepts `acquisition`, `context_modes`,
`acquisition_conditions`, `acquisition_cross_policy`, `save_hidden_states`, and
`save_raw_cache`. The canonical preset is `lmas/secret_digit/preflight.yaml`; it requires
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

### LatentMAS (`configs/lmas/reprop/lm_q38_gsm8k.yaml`)
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
tracking_experiment_name: latentmas-reprop
```

### TextMAS and baseline (`configs/lmas/reprop/bs_q38_gsm8k.yaml`)
```yaml
method: text_mas
model_name: Qwen/Qwen3-0.6B
task: gsm8k
prompt: sequential
max_samples: 5
generate_bs: 1
max_new_tokens: 256
temperature: 0.0
seed: 42
```

### Baseline (`configs/bs_q30.6_gsm8k.yaml`)
```yaml
method: baseline
model_name: Qwen/Qwen3-0.6B
task: gsm8k
max_samples: 5
generate_bs: 1
max_new_tokens: 256
temperature: 0.0
seed: 42
```

---

## 4. Creating a New Configuration

To create a new preset:
1. Create a file inside `configs/` following the naming convention, e.g. `configs/lm_q38_aime2024.yaml`.
2. Define the desired hyperparameters in YAML format.
3. Run directly:
   ```bash
   just run -c lm_q38_aime2024
   ```
