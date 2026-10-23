"""Migrate legacy v1 storage to unified v2 architecture."""

import argparse
import contextlib
import hashlib
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import torch

from .schemas import (
    InventoryItem,
    MigrationItemType,
    MigrationRecord,
    MigrationStatus,
)


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


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
        from .inventory import generate_inventory

        print(f"[*] Scanning source: {self.source_root}")
        report = generate_inventory(self.source_root)
        print(
            f"[*] Found {report.total_files} files ({report.total_bytes / (1024 * 1024):.2f} MB)"
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
            self._copy_sqlite_db(src, dest, item)
        elif item.item_type == MigrationItemType.MLFLOW_ARTIFACT:
            # Preserve artifact relative path structure under mlflow_artifacts/
            # Example: mlflow_artifacts/1/a1b2c3/artifacts/...
            rel = Path(item.rel_path)
            parts = rel.parts
            if "mlflow_artifacts" in parts:
                idx = parts.index("mlflow_artifacts")
                sub = Path(*parts[idx + 1 :])
            else:
                sub = rel
            dest = self.dest_root / "mlflow" / "mlflow_artifacts" / sub
            self._copy_file(src, dest, item)
        elif item.item_type == MigrationItemType.BENCHMARK_RUN:
            rel = Path(item.rel_path)
            parts = rel.parts
            if "runs" in parts:
                idx = parts.index("runs")
                sub = Path(*parts[idx:])
            else:
                sub = rel
            dest = self.dest_root / "evaluation" / sub
            self._copy_file(src, dest, item)
        elif item.item_type == MigrationItemType.TENSOR_BINARY:
            self._migrate_tensor(src, item)
        else:
            # Other files (acquisition summaries, etc.)
            dest = self.dest_root / "artifacts" / item.rel_path
            self._copy_file(src, dest, item)

    def _copy_sqlite_db(self, src: Path, dest: Path, item: InventoryItem) -> None:
        if self.dry_run:
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=None,
                    checksum_source=item.checksum or "",
                    checksum_dest=None,
                    status=MigrationStatus.PENDING,
                )
            )
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Idempotent check: if destination DB exists and has readable runs table
        if dest.is_file():
            try:
                test_conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
                test_cur = test_conn.cursor()
                test_cur.execute("SELECT COUNT(*) FROM runs")
                test_conn.close()
                self.records.append(
                    MigrationRecord(
                        source_path=str(src),
                        destination_path=str(dest),
                        item_type=item.item_type,
                        run_id=None,
                        checksum_source=item.checksum or "",
                        checksum_dest=compute_sha256(dest),
                        status=MigrationStatus.SKIPPED,
                    )
                )
                return
            except Exception:
                pass

        # Perform atomic online SQLite backup
        tmp_dest = dest.with_suffix(".tmp")
        try:
            src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            dest_conn = sqlite3.connect(str(tmp_dest))
            src_conn.backup(dest_conn)
            src_conn.close()
            dest_conn.close()

            # Fix artifact URIs in the copied DB to point to destination artifact store
            dest_conn = sqlite3.connect(str(tmp_dest))
            cur = dest_conn.cursor()
            new_artifact_root = (
                self.dest_root / "mlflow" / "mlflow_artifacts"
            ).as_uri()
            cur.execute("SELECT run_uuid, artifact_uri FROM runs")
            rows = cur.fetchall()
            for r_id, old_uri in rows:
                if old_uri and ("mlflow_artifacts" in old_uri or ".cache" in old_uri):
                    parts = old_uri.split("mlflow_artifacts/")
                    if len(parts) > 1:
                        updated_uri = f"{new_artifact_root}/{parts[1]}"
                        cur.execute(
                            "UPDATE runs SET artifact_uri = ? WHERE run_uuid = ?",
                            (updated_uri, r_id),
                        )
            dest_conn.commit()
            dest_conn.close()

            tmp_dest.replace(dest)
            dest_checksum = compute_sha256(dest)
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=None,
                    checksum_source=item.checksum or "",
                    checksum_dest=dest_checksum,
                    status=MigrationStatus.COPIED,
                )
            )
        except Exception as exc:
            if tmp_dest.exists():
                tmp_dest.unlink(missing_ok=True)
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=None,
                    checksum_source=item.checksum or "",
                    checksum_dest=None,
                    status=MigrationStatus.FAILED,
                    error=str(exc),
                )
            )

    def _copy_file(self, src: Path, dest: Path, item: InventoryItem) -> None:
        if self.dry_run:
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=item.run_id,
                    checksum_source=item.checksum or "",
                    checksum_dest=None,
                    status=MigrationStatus.PENDING,
                )
            )
            return

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Idempotent skip
        if dest.is_file() and compute_sha256(dest) == item.checksum:
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=item.run_id,
                    checksum_source=item.checksum or "",
                    checksum_dest=item.checksum,
                    status=MigrationStatus.SKIPPED,
                )
            )
            return

        tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")
        try:
            shutil.copy2(src, tmp_dest)
            tmp_dest.replace(dest)
            dest_checksum = compute_sha256(dest)
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=item.run_id,
                    checksum_source=item.checksum or "",
                    checksum_dest=dest_checksum,
                    status=MigrationStatus.COPIED,
                )
            )
        except Exception as exc:
            if tmp_dest.exists():
                tmp_dest.unlink(missing_ok=True)
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=item.run_id,
                    checksum_source=item.checksum or "",
                    checksum_dest=None,
                    status=MigrationStatus.FAILED,
                    error=str(exc),
                )
            )

    def _migrate_tensor(self, src: Path, item: InventoryItem) -> None:
        run_id = item.run_id or "orphans"
        artifact_id = src.stem

        dest_dir = self.dest_root / "tensors" / run_id / "run" / "main"
        dest = dest_dir / f"0_{src.name}"

        if self.dry_run:
            self.records.append(
                MigrationRecord(
                    source_path=str(src),
                    destination_path=str(dest),
                    item_type=item.item_type,
                    run_id=run_id,
                    checksum_source=item.checksum or "",
                    checksum_dest=None,
                    status=MigrationStatus.PENDING,
                )
            )
            return

        self._copy_file(src, dest, item)

        # Update tensor_manifest.json for this run
        try:
            tensor = torch.load(dest, map_location="cpu", weights_only=False)
            shape = list(tensor.shape) if hasattr(tensor, "shape") else []
            dtype = str(getattr(tensor, "dtype", "unknown"))
        except Exception:
            shape = []
            dtype = "unknown"

        manifest_path = self.dest_root / "tensors" / run_id / "tensor_manifest.json"
        manifest_data = {"run_id": run_id, "schema_version": "1.0.0", "entries": {}}
        if manifest_path.is_file():
            with contextlib.suppress(Exception):
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        entry_data = {
            "run_id": run_id,
            "sample_key": "run",
            "branch_id": "main",
            "segment_id": "0",
            "artifact_id": artifact_id,
            "artifact_type": "migrated_tensor",
            "storage_uri": f"file://{dest.resolve().as_posix()}",
            "shape": shape,
            "dtype": dtype,
            "checksum": item.checksum or compute_sha256(dest),
            "schema_version": "1.0.0",
            "retention_class": "persistent",
            "status": "complete",
            "metadata": {"migrated_from": str(src)},
        }
        comp_key = f"run:main:0:{artifact_id}"
        manifest_data["entries"][comp_key] = entry_data
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

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
