# Storage, Cache & Path Management

This project enforces strict physical separation and lifecycle management of all file paths, persistent research data, execution caches, and tracking artifacts.

---

## 1. Storage Architecture Overview

The repository partitions data across four distinct lifecycle tiers:

```text
Repository Root
├── storage/                    # Tier 1: Persistent Research Data & Tracking (Permanent)
│   ├── mlflow/                 # MLflow SQLite DB (mlflow.db) & Run Artifacts (mlflow_artifacts/)
│   └── experiments/            # File-based execution runs (config.json, samples/<key>/<cond>/result.json)
│       └── <run_id>/
│           ├── config.json
│           └── samples/
│               └── <sample_key>/
│                   ├── shared/
│                   └── <condition>/
│                       ├── hidden.pt
│                       └── result.json
│
├── artifacts/                  # Tier 2: Publication & Presentation Artifacts (Permanent)
│   ├── presentation/           # High-resolution publication plots (PNG/PDF)
│   ├── receiver_acquisition/   # Analysis tables and conditions
│   └── receiver_compute_preflight/ # Budget curves and model reports
│
├── .cache/                     # Tier 3: Volatile Execution Cache (Safely Cleanable)
│   ├── models/realign/         # Reconstructable projection matrices
│   ├── evaluation/runs/        # Intermediate run checkpoints & sample logs
│   └── runtime/                # Scratchpad JSONL & temporary execution files
│
└── data/                       # Tier 4: Immutable Benchmark Datasets (Read-Only)
    └── medqa.json
```

---

## 2. Path Resolution (`PathResolver`)

All relative and project-specific filesystem paths are resolved centrally through [`PathResolver`](file:///home/user3/projects/latentmas-storage/src/latentmas_reprop/infrastructure/paths/resolver.py).

### Key Managed Locations:
- **`root`**: Root directory of the repository (anchored at `pyproject.toml`). Overridable via `LATENTMAS_PROJECT_ROOT`.
- **`storage_dir`**: Persistent research data directory (`<repo_root>/storage`). Overridable via `LATENTMAS_STORAGE_DIR`.
- **`experiments_dir`**: File-based experiment runs directory (`<storage_dir>/experiments`).
- **`mlflow_dir`**: Persistent MLflow database & artifacts (`<storage_dir>/mlflow`). Overridable via `LATENTMAS_MLFLOW_DIR`.
- **`artifacts_dir`**: Analysis, report, and publication assets (`<repo_root>/artifacts`). Overridable via `LATENTMAS_ARTIFACTS_DIR`.
- **`cache_dir`**: Volatile execution cache (`<repo_root>/.cache`). Overridable via `LATENTMAS_CACHE_DIR`.
- **`data_dir`**: Immutable problem datasets (`<repo_root>/data`).
- **`configs_dir`**: Presets and experiment YAML configs (`<repo_root>/configs`).

### Resolving Configs:
When `-c <name>` is provided to the CLI, `PathResolver.resolve_config_path()` resolves the file in the following order:
1. Exact file path or path relative to the current working directory.
2. Inside `<repo_root>/configs/`.
3. Inside `<repo_root>/configs/` with `.yaml` appended.
4. Inside `<repo_root>/configs/` with `.yml` appended.

---

## 3. File-Based Execution Storage (`FileStorage`)

Experiment states and tensors are stored directly on the filesystem organized by `run_id`, `sample_key`, and `condition`:

- **Storage Location**: `<storage_dir>/experiments/<run_id>/samples/<sample_key>/<condition>/`
- **Completion Check**: `is_condition_complete(condition_dir)` verifies that `result.json` parses as valid JSON and all files listed in its `"files"` field exist on disk.
- **Atomic Writes**: Tensors and `result.json` are written to temporary files and atomically renamed to prevent corruption.
- **Config Verification**: Resuming verifies that the requested configuration matches `<run_id>/config.json`.

---

## 4. Execution Cache & Safe Cleanup

Volatile intermediate caches are stored under `.cache/`.

### Safe Cache Cleanup:
To clean volatile run caches, scratch files, and realignment matrices **without** endangering the MLflow database or persistent research experiments:

```bash
just clean-cache
```

This invokes `ExecutionCacheManager.clean_volatile_cache()`, which:
1. Deletes pure scratch layers (`runtime/`, `models/realign/`, etc.).
2. Validates paths with `PathResolver.is_safe_cache_path()` to ensure protected directories (`storage/`, `data/`, `artifacts/`, `configs/`) are never deleted or affected.

---

## 5. Experiment Tracking with MLflow

MLflow tracks experiment metrics, parameters, and GenAI traces:
- **Backend Store URI**: `sqlite:///<storage_dir>/mlflow/mlflow.db`
- **Default Artifact Root**: `<storage_dir>/mlflow/mlflow_artifacts`
- **Data Link**: Each MLflow run tags `data_dir` pointing to the run's directory in `storage/experiments/<run_id>`.
