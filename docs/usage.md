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

Recompute preflight statistics directly from saved MLflow artifacts with
`just analyze-receiver`. No inference runs and no generated report is used as input.
Receiver observations come from `results/sample_results.jsonl`; Sender cells come
from `probe/results.json` and `probe/null_statistics.json` in
`latentmas_sender_probe`, linked to acquisition metadata by `source_run_id`.
Downloads are cached under `.cache/receiver_acquisition/mlflow/<run_id>/`.

The output directory contains `summary.md`, run-specific `diagnostics.json`, and
CSV/JSON tables: `sender_probe_cells`, `receiver_conditions`, `receiver_per_digit`,
`receiver_dependencies`, and `model_steps_summary`. Sender 20-step trajectories
and 1-/4-step comparisons are separate. Sender null means and raw/max-statistic
p-values are recomputed from saved permutation accuracies; each run's complete
step-by-representation set forms its FWER family. Missing permutations produce
missing p-values. `--sender-run-id` selects among ambiguous saved probe runs.

Receiver tables include source/prediction counts, raw and balanced accuracy,
recall, confusion matrices, prediction entropy, paired changes, digit probability
mass and normalized distributions, Jensen-Shannon divergence, NLL, and Brier.
Balanced accuracy averages recalls over source digits present in the condition.
Entropy, MI, JS, and NLL use natural logarithms. NLL and multiclass Brier use
probabilities normalized within digits 0–9. Matched Drop scores evaluate the
same sample's saved Drop prediction against that condition's source label,
including Cross labels; paired deltas always use that same label. Missing saved
values remain null in JSON, empty in CSV, and `—` in Markdown.

Dependency tests shuffle source labels against fixed predictions within each
condition. `--permutations` (default 5000) and `--seed` (default 0) control this
reproducible analysis. Raw permutation MI arrays are retained for further
correction; Holm p-values cover all selected Receiver Own/Cross dependency tests.
No acquisition, communication, or other interpretation is assigned.

Only complete 100-sample runs are selected automatically. When multiple runs
qualify for the same model and latent step, the command stops with their run IDs;
specify the intended run using `--canonical-run-id RUN_ID` (repeat for each
ambiguous model/step). An MLflow `receiver_analysis_canonical=true` tag also
designates a run. The report's coverage table lists each selected run ID and
sample count.
For the current saved Receiver set, the explicitly selected latest 4-step runs are:

```bash
just analyze-receiver \
  --canonical-run-id 4d3b741b0d1f4a569da95c0474097ad0 \
  --canonical-run-id 589f77a1f63241459d6ea902337fcea7
```

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

### Additional saved-feature diagnostics

Receiver source-label permutation tests additionally retain raw null arrays for
normalized digit NLL and Brier (lower-tail tests) and balanced accuracy (upper-tail
test). Predictions and probability vectors are fixed; labels are shuffled within
each run and condition, preserving source counts. Missing candidate losses leave
the corresponding soft test missing. These tests share `--permutations` and
`--seed` with MI but use independent RNG streams (1 for balanced accuracy, 2 for
soft losses). NLL and Brier share each soft-label shuffle. The Monte Carlo p-value
uses `(extreme + 1) / (permutations + 1)` with floating-point tie tolerance, following
[the randomized permutation convention](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).
Holm FWER corrections cover all selected Own/Cross conditions, separately for MI,
NLL, Brier, and balanced accuracy. Drop statistics are reported separately from
these families.

`receiver_inference_tests.csv/json` retains these tests and raw null arrays.
`receiver_probability_samples.csv/json` records correct-digit probability, its
digit-normalized probability, rank, top-2/top-3 membership, margin
`q_correct - max(q_wrong)`, NLL, and Brier for each sample. Rank ties use ascending
digit order. `receiver_per_digit` additionally reports source-specific means and
probability coverage; absent source digits remain missing.

When `probe/sender_latent_states.pt` exists on the linked acquisition run, Sender
analysis also writes `sender_robustness.csv/json` and `sender_geometry.csv/json`.
Robustness uses the saved template-grouped folds and reports every preset without
selecting a best configuration: standardized Ridge with alpha
`0.01, 0.1, 1, 10, 100`; logistic regression with and without standardization
(`C=1`, `max_iter=1000`, `tol=1e-4`); standardized LDA (`solver=svd`);
and standardized 5-neighbor kNN (Euclidean distance, uniform weights).
StandardScaler is fitted only on training rows. Fold convergence status and OOF
predictions are retained; these robustness results do not receive new permutation
p-values. LDA uses the
[SVD solver](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html).

Sender effect intervals are 95% percentile template-cluster bootstrap intervals
on saved OOF correctness minus the fixed permutation-null mean, using
`--permutations` bootstrap draws and `--seed`. They condition on the saved fitted
probes and do not include retraining or null-mean estimation uncertainty.
Geometry uses original, unstandardized features: class-centroid distances,
unscaled within/between-class sums of squares, and the compact covariance
spectrum (centered features, denominator `n-1`; omitted feature-space eigenvalues
are zero). Effective rank is `exp(entropy(normalized positive eigenvalues))`;
participation ratio is `sum(eigenvalues)^2 / sum(eigenvalues^2)`. Eigenvalues below
`max_eigenvalue * 1e-12` are excluded from numerical rank and these rank metrics.
No model inference or automatic communication/acquisition interpretation occurs.
