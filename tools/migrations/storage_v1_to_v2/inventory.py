"""Scan and report inventory of legacy v1 storage files."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from .schemas import InventoryItem, InventoryReport, MigrationItemType


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_sqlite_runs(db_path: Path) -> dict[str, dict[str, str]]:
    """Extract known run metadata from legacy MLflow SQLite DB."""
    runs = {}
    if not db_path.is_file():
        return runs
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT run_uuid, experiment_id, name, status, artifact_uri FROM runs"
        )
        for row in cursor.fetchall():
            run_uuid, exp_id, name, status, artifact_uri = row
            runs[run_uuid] = {
                "experiment_id": str(exp_id),
                "name": str(name or ""),
                "status": str(status or ""),
                "artifact_uri": str(artifact_uri or ""),
            }
        conn.close()
    except Exception as exc:
        print(f"[Warning] Failed to query SQLite DB {db_path}: {exc}", file=sys.stderr)
    return runs


def generate_inventory(source_root: Path) -> InventoryReport:
    """Scan legacy storage directory and build an InventoryReport."""
    source_root = source_root.resolve()
    items: list[InventoryItem] = []
    run_ids: set[str] = set()

    # 1. Search for MLflow DB
    db_candidates = [
        source_root / "mlflow.db",
        source_root / ".cache" / "mlflow.db",
        source_root / "mlflow" / "mlflow.db",
        source_root / ".cache" / "mlflow" / "mlflow.db",
    ]
    db_path = next((p for p in db_candidates if p.is_file()), None)
    known_runs = scan_sqlite_runs(db_path) if db_path else {}
    run_ids.update(known_runs.keys())

    if db_path:
        items.append(
            InventoryItem(
                rel_path=str(db_path.relative_to(source_root)),
                abs_path=str(db_path),
                item_type=MigrationItemType.MLFLOW_DB,
                size_bytes=db_path.stat().st_size,
                checksum=compute_sha256(db_path),
                notes=f"MLflow SQLite DB with {len(known_runs)} runs",
            )
        )

    # 2. Walk directory for all other files
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        if db_path and path.resolve() == db_path.resolve():
            continue

        rel = str(path.relative_to(source_root))
        size = path.stat().st_size
        checksum = compute_sha256(path)

        # Classify item type and check run association
        matched_run = None
        for r_id in known_runs:
            if r_id in rel:
                matched_run = r_id
                break

        item_type = MigrationItemType.OTHER
        is_orphan = False

        if "mlflow_artifacts" in rel or (matched_run and "artifacts" in rel):
            item_type = MigrationItemType.MLFLOW_ARTIFACT
        elif "evaluation/runs" in rel:
            item_type = MigrationItemType.BENCHMARK_RUN
            # Check if directory name is run_id
            parts = Path(rel).parts
            if "runs" in parts:
                idx = parts.index("runs")
                if idx + 1 < len(parts):
                    run_candidate = parts[idx + 1]
                    matched_run = run_candidate
                    run_ids.add(run_candidate)
        elif "receiver_acquisition" in rel:
            item_type = MigrationItemType.ACQUISITION_RESULT
            if path.suffix == ".pt":
                item_type = MigrationItemType.TENSOR_BINARY
            if not matched_run and path.name in {
                "sample_results.json",
                "summary.json",
                "sender_latent_states.pt",
            }:
                is_orphan = True
        elif "latent_interventions" in rel or "interventions" in rel:
            item_type = MigrationItemType.INTERVENTION_KV
            if not matched_run:
                is_orphan = True
        elif path.suffix == ".pt":
            item_type = MigrationItemType.TENSOR_BINARY
            if not matched_run:
                is_orphan = True

        items.append(
            InventoryItem(
                rel_path=rel,
                abs_path=str(path),
                item_type=item_type,
                size_bytes=size,
                run_id=matched_run,
                checksum=checksum,
                is_orphan=is_orphan,
                notes="Orphaned legacy file without clear run attribution"
                if is_orphan
                else "",
            )
        )

    total_bytes = sum(i.size_bytes for i in items)
    orphan_count = sum(1 for i in items if i.is_orphan)

    return InventoryReport(
        source_root=str(source_root),
        total_files=len(items),
        total_bytes=total_bytes,
        run_count=len(run_ids),
        orphan_count=orphan_count,
        items=items,
    )


def print_summary(report: InventoryReport) -> None:
    print("\n=== Legacy Storage Inventory Summary ===")
    print(f"Source Root:   {report.source_root}")
    print(f"Total Files:   {report.total_files}")
    print(f"Total Volume:  {report.total_bytes / (1024 * 1024):.2f} MB")
    print(f"Known Runs:    {report.run_count}")
    print(f"Orphan Files:  {report.orphan_count}")
    print("========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory legacy v1 storage files.")
    parser.add_argument(
        "--source", required=True, help="Path to legacy storage root (e.g. .cache)"
    )
    parser.add_argument("--output", help="Optional output JSON file path")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Error: Source directory '{source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    report = generate_inventory(source)
    print_summary(report)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Detailed inventory report saved to: {out_path}")
    else:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
