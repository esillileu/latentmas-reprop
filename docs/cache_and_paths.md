# Cache & Path Management

This project enforces strict management of all file paths and cache artifacts to avoid hardcoded paths and arbitrary disk writes.

---

## 1. Path Resolution (`PathResolver`)

All relative and project-specific filesystem paths are resolved centrally through [`PathResolver`](file:///home/esillileu/projects/latentmas-reprop/src/latentmas_reprop/infrastructure/paths/resolver.py).

### Key Managed Locations:
- **`root`**: Root directory of the repository (anchored at `pyproject.toml`).
- **`configs_dir`**: Presets directory (`<repo_root>/configs`).
- **`cache_dir`**: Root cache directory (`<repo_root>/.cache`).
- **`data_dir`**: Local dataset files (`<repo_root>/data`).
- **`medqa_json_path`**: Location of MedQA dataset (`<repo_root>/data/medqa.json`).

### Resolving Configs:
When `-c <name>` is provided to the CLI, `PathResolver.resolve_config_path()` resolves the file in the following order:
1. Exact file path or path relative to the current working directory.
2. Inside `<repo_root>/configs/`.
3. Inside `<repo_root>/configs/` with `.yaml` appended.
4. Inside `<repo_root>/configs/` with `.yml` appended.

---

## 2. Unified Cache Hierarchy (`.cache/`)

Non-HuggingFace caches are strictly partitioned under the repository root `.cache/`:

```
.cache/
├── models/
│   └── realign/           # Precomputed latent realignment projection matrices
│       └── <model_hash>_realign.pt
└── evaluation/
    └── runs/              # JSON logs of benchmark execution runs
        └── <run_timestamp>_<method>_<task>_<model>.json
```

### 2.1 Latent Realignment Cache (`.cache/models/realign/`)
When `--latent_space_realign` is used, the alignment projection matrix computed between latent states and token embeddings is cached to disk so subsequent runs reuse it instantly without recalculation.

### 2.2 Evaluation Runs Cache (`.cache/evaluation/runs/`)
Every benchmark execution produces a structured JSON artifact recording:
- Experiment configuration and timestamp
- Aggregated accuracy, sample counts, and runtime
- Per-sample predictions, gold answers, and complete multi-agent traces

### 2.3 Cleaning Caches
To remove non-HuggingFace run caches and realignment matrices:
```bash
just clean-cache
```
Or programmatically via `ExecutionCacheManager`:
```python
from latentmas_reprop.infrastructure.cache.manager import get_cache_manager

cache = get_cache_manager()
cache.clear_all()
```
