"""Migrate legacy v1 storage to unified v2 architecture."""

import argparse
import json
import sys
from pathlib import Path

from .inventory import generate_inventory
from .migrator_ops import (
    compute_sha256,
    copy_file,
    copy_sqlite_db,
    migrate_tensor,
)

__all__ = ["StorageMigrator", "compute_sha256", "main"]
from .schemas import (
    InventoryItem,
    MigrationItemType,
    MigrationRecord,
    MigrationStatus,
)


class StorageMigrator:
    """Safely migrates legacy data artifacts into unified v2 storage layout."""

    def __init__(
        self, source_root: Path, dest_root: Path, dry_run: bool = True
    ) -> None:
        self.source_root = source_root.resolve()
        self.dest_root = dest_root.resolve()
        self.dry_run = dry_run
        self.records: list[MigrationRecord] = []

    def run(self) -> list[MigrationRecord]:
        """Execute migration plan."""
        print(f"[*] Scanning source: {self.source_root}")
        report = generate_inventory(self.source_root)
        print(
            f"[*] Found {report.total_files} files "
            f"({report.total_bytes / (1024 * 1024):.2f} MB)"
        )

        if self.dry_run:
            print("\n[DRY RUN MODE] No changes will be made to destination.")
        else:
            print("\n[EXECUTE MODE] Copying and converting data to destination...")
            self.dest_root.mkdir(parents=True, exist_ok=True)

        for item in report.items:
            self._migrate_item(item)

        if not self.dry_run:
            self._save_migration_manifest()

        return self.records

    def _migrate_item(self, item: InventoryItem) -> None:
        src = Path(item.abs_path)

        if item.item_type == MigrationItemType.MLFLOW_DB:
            dest = self.dest_root / "mlflow" / "mlflow.db"
            rec = copy_sqlite_db(src, dest, item, self.dest_root, self.dry_run)
            self.records.append(rec)
        elif item.item_type == MigrationItemType.MLFLOW_ARTIFACT:
            rel = Path(item.rel_path)
            parts = rel.parts
            if "mlflow_artifacts" in parts:
                idx = parts.index("mlflow_artifacts")
                sub = Path(*parts[idx + 1 :])
            else:
                sub = rel
            dest = self.dest_root / "mlflow" / "mlflow_artifacts" / sub
            rec = copy_file(src, dest, item, self.dry_run)
            self.records.append(rec)
        elif item.item_type == MigrationItemType.BENCHMARK_RUN:
            rel = Path(item.rel_path)
            parts = rel.parts
            if "runs" in parts:
                idx = parts.index("runs")
                sub = Path(*parts[idx + 1 :])
            else:
                sub = rel
            dest = self.dest_root / "experiments" / sub
            rec = copy_file(src, dest, item, self.dry_run)
            self.records.append(rec)
            if (
                not self.dry_run
                and rec.status in {MigrationStatus.COPIED, MigrationStatus.SKIPPED}
                and src.name == "samples.jsonl"
            ):
                self._migrate_benchmark_samples(dest)
        elif item.item_type == MigrationItemType.TENSOR_BINARY:
            rec = migrate_tensor(src, item, self.dest_root, self.dry_run)
            self.records.append(rec)
        else:
            dest = self.dest_root / "artifacts" / item.rel_path
            rec = copy_file(src, dest, item, self.dry_run)
            self.records.append(rec)

    def _migrate_benchmark_samples(self, dest: Path) -> None:
        try:
            with open(dest, encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if "sample_index" in row:
                        skey = f"{int(row['sample_index']):06d}"
                    elif "sample_key" in row:
                        sk = str(row["sample_key"])
                        skey = f"{int(sk):06d}" if sk.isdigit() else sk
                    else:
                        skey = f"{line_idx:06d}"
                    sample_dir = dest.parent / "samples" / skey / "benchmark"
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    res_file = sample_dir / "result.json"
                    if not res_file.exists():
                        tmp_file = res_file.with_suffix(".tmp")
                        with open(tmp_file, "w", encoding="utf-8") as out_f:
                            json.dump(row, out_f, indent=2)
                        tmp_file.replace(res_file)
        except Exception:
            pass

    def _save_migration_manifest(self) -> None:
        manifest_file = self.dest_root / "migration_manifest.json"
        data = {
            "source_root": str(self.source_root),
            "destination_root": str(self.dest_root),
            "total_items": len(self.records),
            "copied": sum(
                1 for r in self.records if r.status == MigrationStatus.COPIED
            ),
            "skipped": sum(
                1 for r in self.records if r.status == MigrationStatus.SKIPPED
            ),
            "failed": sum(
                1 for r in self.records if r.status == MigrationStatus.FAILED
            ),
            "records": [r.to_dict() for r in self.records],
        }
        manifest_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[*] Migration manifest written to: {manifest_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate legacy v1 storage files to unified v2 layout."
    )
    parser.add_argument(
        "--source", required=True, help="Path to legacy storage root (e.g. .cache)"
    )
    parser.add_argument(
        "--destination", required=True, help="Path to new storage root (e.g. storage)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform actual migration (default is dry-run)",
    )
    parser.add_argument(
        "--report", help="Optional path to output migration execution report JSON"
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.destination)

    if not source.exists():
        print(f"Error: Source directory '{source}' does not exist.", file=sys.stderr)
        sys.exit(1)

    migrator = StorageMigrator(
        source_root=source, dest_root=dest, dry_run=not args.execute
    )
    records = migrator.run()

    copied = sum(1 for r in records if r.status == MigrationStatus.COPIED)
    skipped = sum(1 for r in records if r.status == MigrationStatus.SKIPPED)
    failed = sum(1 for r in records if r.status == MigrationStatus.FAILED)
    pending = sum(1 for r in records if r.status == MigrationStatus.PENDING)

    print("\n=== Migration Result Summary ===")
    if not args.execute:
        print(
            f"Plan: {pending} items queued for migration. Run with --execute to perform migration."
        )
    else:
        print(f"Copied:  {copied}")
        print(f"Skipped: {skipped} (already migrated)")
        print(f"Failed:  {failed}")
    print("================================\n")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_data = [r.to_dict() for r in records]
        report_path.write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Migration records saved to: {report_path}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
