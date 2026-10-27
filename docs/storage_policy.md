# Storage Policy — Deterministic File-Based Execution Model

- **Architecture**: Deterministic Human-Readable Experiment Directories & File-Based Condition Execution
- **Version**: 2.1.0
- **Status**: Implemented (`ref/storage-policy`)

---

## 1. Core Principles

1. **The directory path itself is the sole external identifier of an experiment.**
2. Identical calculation configurations always map to the exact same directory.
3. Different calculation configurations map to different directories.
4. Completed calculations are never re-executed.
5. Re-executing an interrupted command resumes only uncompleted calculations.
6. MLflow tracks experiments but does **not** decide whether a calculation is complete.
7. Downstream analyzers query experiment data via MLflow / linked local storage.
8. Analysis results and reports are preserved both locally in `artifacts/` and in MLflow under `latentmas-analysis`.

### Prohibited Patterns
- No UUID-based execution identification.
- No separate identifier management systems (Config ID, storage Run ID, Sample Key).
- No configuration hashes in directory names.
- No external manifest or database for run status.
- No checkpoint JSONL as a completion indicator.
- No local storage vs. MLflow status synchronization managers or adapters.

---

## 2. Storage Tiers & Directory Layout

The workspace root adheres to a clean, minimal layout:

```text
latentmas-reprop/
├── configs/
│   └── <preset YAML configurations>
├── data/
│   └── <input benchmark datasets>
├── storage/
│   ├── experiments/
│   │   └── <task>/<model>/<method>/<prompt>/<params>/
│   │       ├── config.json
│   │       ├── summary.json
│   │       ├── samples.jsonl (generated for export/MLflow)
│   │       └── samples/
│   │           ├── 000000/
│   │           │   ├── shared/
│   │           │   │   ├── kv.pt
│   │           │   │   └── result.json
│   │           │   └── <condition>/
│   │           │       └── result.json
│   │           └── 000001/
│   │               └── ...
│   └── mlflow/
│       ├── mlflow.db
│       └── mlflow_artifacts/
├── artifacts/
│   └── <human-readable analysis path>/
│       ├── summary.md
│       └── <plots and CSV tables>
└── .cache/
    └── <reconstructable volatile caches, cleanable via just clean-cache>
```

### Storage Directory Roles:
1. **`configs/`**: User-defined YAML preset configurations and CLI overrides.
2. **`data/`**: Reference datasets (GSM8K, MedQA, etc.).
3. **`storage/experiments/`**: Authoritative source of truth for all computation outputs, sample-level records, and large tensors.
4. **`storage/mlflow/`**: Persistent SQLite tracking database (`mlflow.db`) and lightweight tracked artifacts (`mlflow_artifacts/`).
5. **`artifacts/`**: Final analysis reports, figures, and publication tables.
6. **`.cache/`**: Volatile, cleanable cache (e.g. projection weights, tokenizers). Cleaned safely at any time via `just clean-cache` without affecting research data.

---

## 3. Path Construction Contract

Experiment paths are canonical 5-level relative paths constructed before model loading:

$$\text{path} = \langle\text{task}\rangle / \langle\text{clean\_model\_name}\rangle / \langle\text{method}\rangle / \langle\text{prompt}\rangle / \langle\text{params\_stem}\rangle$$

Examples:
- **Benchmark**: `gsm8k/Qwen3-8B/latent_mas/sequential/ls20_seed42_temp0.6_top0.95_mt2048_float16`
- **Intervention**: `gsm8k/Qwen3-8B/latent_mas/sequential/ls20_seed42_temp0.6_top0.95_mt2048_float16_intervention_cross_shift_1`
- **Acquisition**: `secret_digit/Qwen3-0.6B/latent_mas/default/ls4_seed42_temp0.6_top0.95_mt2048_float16_acquisition_modes_full-latent_only-prompt_only_cond_drop-own-cross`

### Path Rules:
- The path is decided **before** model loading using execution arguments and explicit or resolved precision (`resolve_experiment_dtype_name`).
- Runner, UseCase, Storage, MLflow, and Reader all reference the exact same relative directory path.
- The path cannot change dynamically mid-execution.

---

## 4. Completion & Resumption Contracts

### 4.1 Condition-Level Completion Standard
Each condition directory's `result.json` is the sole authoritative standard of computation completion:

| File Status | Completion Decision | Action |
| :--- | :--- | :--- |
| `result.json` missing | Incomplete | Execute this condition |
| `result.json` valid JSON, no error, all required files exist | Complete | Skip execution, reuse stored result |
| `result.json` corrupted or error present | Incomplete | Recompute this condition |
| `.tmp` files present | Incomplete task | Clean up and recompute |

### 4.2 Experiment Completion Standard
An experiment execution is complete if and only if **all required samples and conditions** have valid completion files on disk:
- `summary.json`: Aggregated metrics and metadata.
- `config.json`: Recorded execution configuration.
- `samples.jsonl`: Generated from sample directories for export and MLflow.
- The mere existence of `summary.json` or `parity.json` does **not** mark an experiment as complete if sample condition files are missing or incomplete.
- Re-executing an experiment with a larger `max_samples` reuses completed samples (e.g. 0 to 9) and computes only newly requested samples (e.g. 10 to 49).
- Supplying `--overwrite` forces re-computation of the requested range only.

---

## 5. MLflow Tracking Boundary

Each execution run in MLflow corresponds directly to one experiment directory path:
- **MLflow Tags**:
  - `data_dir`: Absolute path to `storage/experiments/<relpath>/`
  - `rel_dir`: Canonical relative path `<task>/<model>/<method>/<prompt>/<params>`
- **Run Idempotency**:
  - If a directory is re-executed, the existing MLflow run matching `tags.data_dir` or `tags.rel_dir` is reused. No duplicate runs are created.
- **Tracked in MLflow**: Configuration, git commit hash, summary metrics, GenAI traces, evaluation assessments, and lightweight result files.
- **Retained on Disk**: Heavy tensors (KV caches, full hidden states) remain in `storage/experiments/`.

---

## 6. Analysis Integration Contract

All analysis tools follow a unified workflow:
1. **Input Selection**: Select input experiment directory path(s) (e.g. `--experiment-dir`).
2. **Data Querying**: Query MLflow for matching experiment metadata and retrieve artifacts / tensors via `ResearchDataReader` without requiring MLflow Run IDs in the CLI.
3. **Execution**: Compute analysis metrics and generate figures without model inference.
4. **Persistence**:
   - Write figures and tables to `artifacts/<analysis_path>/`.
   - Record results and upload all outputs to the dedicated MLflow experiment `latentmas-analysis`.
