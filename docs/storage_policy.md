# Storage Policy — File-Based Execution Model

- **Target Architecture**: File-Based Condition Execution & Resumption Model
- **Version**: 2.0.0
- **Status**: Implemented (`ref/storage-policy`)

---

## 1. Storage Tiers & Directory Structure

All experiment outputs and assets adhere to a clear, minimal storage layout without external state databases or continuous manifest synchronization:

```text
storage/
├── mlflow/
│   ├── mlflow.db
│   └── mlflow_artifacts/
│
└── experiments/
    └── <run_id>/
        ├── config.json
        │
        └── samples/
            ├── <sample_key_A>/
            │   ├── shared/
            │   │   ├── hidden.pt
            │   │   ├── kv.pt
            │   │   └── result.json
            │   │
            │   ├── ple/
            │   │   ├── hidden.pt
            │   │   └── result.json
            │   │
            │   ├── plce/
            │   │   ├── hidden.pt
            │   │   └── result.json
            │   │
            │   └── plpe/
            │       ├── hidden.pt
            │       └── result.json
            │
            └── <sample_key_B>/
                └── ...
```

### Storage Directory Roles:
1. **`storage/mlflow/`**: Persistent SQLite tracking database (`mlflow.db`) and lightweight tracked artifacts (`mlflow_artifacts/`).
2. **`storage/experiments/`**: Source of truth for all experiment outputs, sample results, and heavy tensors organized by `run_id`.
3. **`.cache/`**: Volatile, reconstructable cache (e.g., projection weights, tokenizers). Cleanable at any time via `just clean-cache` without affecting research data.
4. **`artifacts/`**: Publication and presentation figures and reports.

---

## 2. Identifier Policy

- **`run_id`**: Directory identifier for a concrete execution run. Distinct executions use different `run_id` values.
- **`sample_key`**: Content-based identifier for a problem or prompt across dataset splits.
- **`condition`**: Experimental condition or branch name (e.g., `shared`, `own`, `cross`, `drop`, `zero`, `ple`, `plce`, `plpe`).

### Resumption Policy:
- Resuming requires explicitly specifying the existing `run_id` (via CLI `--run_id <id>` or explicit parameter). A new run is never silently treated as an automatic resume of a prior run.
- When resuming, `config.json` in `<experiments_dir>/<run_id>/` is verified against the requested configuration. If critical settings differ (e.g. `model_name`, `task`, `method`, `latent_steps`, `seed`), execution is rejected with `ConfigurationMismatchError`.
- MLflow's Run ID serves as the directory ID; when resuming an existing run under a new MLflow tracking run, the previous `run_id` is linked as a tag on the Run.

---

## 3. Completion & Atomicity Contracts

### 3.1 Completion Decision
Each condition directory's `result.json` acts as the authoritative completion indicator:

| State | Action |
| :--- | :--- |
| `result.json` missing | Execute that condition |
| `result.json` valid & all required files exist | Condition complete; skip execution and load result |
| `result.json` corrupted or required files missing | Incomplete; re-execute that condition |
| `.tmp` files only | Incomplete task; re-execute |

> **Important**: The mere existence of an arbitrary file does not indicate completion. Completion is guaranteed only when `result.json` successfully parses as JSON and every file declared in its `"files"` list exists on disk.

### 3.2 Save Sequence
When condition execution finishes, files are committed in strict atomic order:
1. Save hidden states and other required tensors to `.<filename>.tmp`, then atomically rename.
2. Prepare result metadata (predictions, evaluation scores, latency, `"files": [...]`).
3. Verify required files are successfully saved on disk.
4. Write `result.json.tmp`.
5. Atomically rename `result.json.tmp` to `result.json`.

`result.json` is always written last. If any step fails or is interrupted, no completion file is created.

---

## 4. Execution Resumption Workflow

At the start of execution:
- Completed `shared/` contexts are reused across branches.
- Completed conditions within a sample are reused.
- Any incomplete or corrupted conditions are recomputed.
- Interrupted runs resume simply by inspecting directory contents. No separate checkpoint JSONL, DB, or manifest synchronization is required.

---

## 5. MLflow Tracking Boundary

MLflow is restricted purely to experiment tracking:
- **Tracked in MLflow**: Configuration, git commit hash, metrics, GenAI traces, small result summaries, and data directory link (`data_dir` tag).
- **Not Tracked in MLflow**: Full hidden state trajectories, large KV caches, per-file intermediate states, or manifest synchronizations.
