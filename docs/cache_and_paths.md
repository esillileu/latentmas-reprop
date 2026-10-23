# Storage, Cache & Path Management

This project enforces strict physical separation and lifecycle management of all file paths, persistent research data, execution caches, and tracking artifacts.

---

## 1. Storage Architecture Overview

The repository partitions data across four distinct lifecycle tiers:

```
Repository Root
├── storage/                    # Tier 1: Persistent Research Data & Tracking (Permanent)
│   ├── mlflow/                 # MLflow SQLite DB (mlflow.db) & Run Artifacts (mlflow_artifacts/)
│   └── tensors/                # Universal Tensor Store (KV snapshots, hidden states, manifests)
│       └── <run_id>/
│           ├── <artifact_id>.pt
│           └── tensor_manifest.json
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
- **`tensors_dir`**: Heavy tensor storage (`<storage_dir>/tensors`).
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

## 3. Persistent Tensor Store (`TensorStore`)

Large tensors (such as KV cache snapshots and stepwise hidden states) are decoupled from MLflow SQLite tracking to prevent database bloat and performance degradation:

- **Storage Location**: `<storage_dir>/tensors/<run_id>/<artifact_id>.pt`
- **Integrity Validation**: Every tensor write calculates a SHA-256 checksum, recorded in `tensor_manifest.json`. Loading automatically validates file integrity when `verify_checksum=True`.
- **Cross-Branch References**: Multiple branches or runs can reference an immutable tensor snapshot without physical duplication via `TensorStore.register_reference()`.
- **MLflow Tracking**: Lightweight `tensor_manifest.json` metadata is logged to MLflow; heavy tensor binaries remain in local/remote tensor storage.

---

## 4. Execution Cache & Safe Cleanup

Volatile intermediate caches are stored under `.cache/`.

### Safe Cache Cleanup:
To clean volatile run caches, scratch files, and realignment matrices **without** endangering the MLflow database or persistent research tensors:

```bash
just clean-cache
```

This invokes `ExecutionCacheManager.clean_volatile_cache()`, which:
1. Deletes pure scratch layers (`runtime/`, `models/realign/`, etc.).
2. **Protects Resumption State**: Intermediate evaluation run logs (`evaluation/runs/`) containing unfinished `samples.jsonl` are strictly preserved by default to prevent loss of partial benchmark calculations.
3. Validates paths with `PathResolver.is_safe_cache_path()` to ensure protected directories (`storage/`, `data/`, `artifacts/`, `configs/`) are never deleted or affected, even if symlinked.

Programmatic invocation:
```python
from latentmas_reprop.infrastructure.cache.manager import get_cache_manager

cache = get_cache_manager()
# Routine cleanup: preserves in-progress evaluation run checkpoints
cleaned_dirs = cache.clean_volatile_cache()

# Full cleanup including completed runs:
# cleaned_dirs = cache.clean_volatile_cache(include_resumable_runs=True)
```

---

## 5. Experiment Tracking with MLflow

Local MLflow tracking defaults to:
- **Backend Store URI**: `sqlite:///<storage_dir>/mlflow/mlflow.db`
- **Default Artifact Root**: `<storage_dir>/mlflow/mlflow_artifacts`

### Migration from Legacy Layout:
The legacy path layout used `.cache/mlflow.db` and `.cache/mlflow_artifacts/`.
To migrate existing data to the persistent `storage/` directory, use the standalone migration tool:
```bash
python tools/migrations/storage_v1_to_v2/migrate.py --source .cache --destination storage --execute
```

To launch the local MLflow UI:
```bash
just mlflow-ui
```
