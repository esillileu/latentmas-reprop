# Storage v1 to v2 Standalone Migration Tool

This directory contains the standalone data migration tool for converting legacy `latentmas` v1 experimental data (stored across `.cache/mlflow.db`, `.cache/mlflow_artifacts/`, and `.cache/evaluation/`) into the unified v2 storage architecture (`storage/mlflow/`, `storage/tensors/`, etc.).

---

## 1. Architectural Principles

1. **Complete Separation from Runtime**:
   - Runtime operational code (`src/latentmas_reprop`) does NOT contain legacy compatibility fallbacks or auto-detection branches.
   - All historical format conversions and path re-mappings are handled strictly by this standalone tool.
2. **Read-Only on Source**:
   - The source directory (e.g. `.cache/` or legacy backup) is treated as strictly read-only.
   - Files are copied, never moved or deleted.
   - Original data remains intact for auditing or rollback.
3. **Crash-Safe & Idempotent**:
   - Copies use temporary files (`.tmp`) and atomic filesystem renames.
   - SQLite databases are copied via SQLite online backup API to prevent journal corruption.
   - Re-running the migration skips already copied and verified files.
4. **Explicit Execution**:
   - The default mode is `--dry-run`.
   - Real data copying requires the explicit `--execute` flag.

---

## 2. Directory Layout Comparison

```text
Legacy v1 (.cache)                       Unified v2 (storage)
──────────────────                       ────────────────────
.cache/mlflow.db                   ──▶   storage/mlflow/mlflow.db
.cache/mlflow_artifacts/           ──▶   storage/mlflow/mlflow_artifacts/
.cache/evaluation/runs/<run_id>/   ──▶   storage/evaluation/runs/<run_id>/
.cache/evaluation/.../*.pt         ──▶   storage/tensors/<run_id>/<sample>/<branch>/<seg>_<artifact>.pt
                                         (indexed via tensor_manifest.json with SHA-256)
```

---

## 3. Step-by-Step Migration Guide

### Step 1: Inventory Scan
Scan the legacy directory to assess file count, volume, run linkages, and identify orphan files:

```bash
python tools/migrations/storage_v1_to_v2/inventory.py \
  --source /path/to/legacy_dir \
  --output inventory_report.json
```

### Step 2: Dry-Run Migration
Preview all migration actions without writing to disk:

```bash
python tools/migrations/storage_v1_to_v2/migrate.py \
  --source /path/to/legacy_dir \
  --destination /path/to/new_storage_root
```

### Step 3: Migration Execution
Perform the actual migration with the `--execute` flag:

```bash
python tools/migrations/storage_v1_to_v2/migrate.py \
  --source /path/to/legacy_dir \
  --destination /path/to/new_storage_root \
  --execute \
  --report migration_report.json
```

### Step 4: Verification
Verify file counts, SQLite DB readability, tensor manifest registration, and SHA-256 checksums:

```bash
python tools/migrations/storage_v1_to_v2/verify.py \
  --source /path/to/legacy_dir \
  --destination /path/to/new_storage_root \
  --report verification_report.json
```

If any checksum mismatches or missing files are found, `verify.py` exits with status `1`.

---

## 4. Rollback and Safety

If any unexpected issue occurs during cutover:
1. Since the source directory was never altered, the original environment is 100% preserved.
2. Simply point `MLFLOW_TRACKING_URI` and `LATENTMAS_STORAGE_DIR` back to the original backup or legacy location.
