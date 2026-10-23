"""File, database, and tensor operations for storage v1 to v2 migration."""

import hashlib
import shutil
import sqlite3
from pathlib import Path

from .schemas import (
    InventoryItem,
    MigrationRecord,
    MigrationStatus,
)


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_sqlite_db(
    src: Path, dest: Path, item: InventoryItem, dest_root: Path, dry_run: bool
) -> MigrationRecord:
    if dry_run:
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=None,
            checksum_source=item.checksum or "",
            checksum_dest=None,
            status=MigrationStatus.PENDING,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file():
        try:
            test_conn = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
            test_cur = test_conn.cursor()
            test_cur.execute("SELECT COUNT(*) FROM runs")
            test_conn.close()
            return MigrationRecord(
                source_path=str(src),
                destination_path=str(dest),
                item_type=item.item_type,
                run_id=None,
                checksum_source=item.checksum or "",
                checksum_dest=compute_sha256(dest),
                status=MigrationStatus.SKIPPED,
            )
        except Exception:
            pass

    tmp_dest = dest.with_suffix(".tmp")
    try:
        src_conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
        dest_conn = sqlite3.connect(str(tmp_dest))
        src_conn.backup(dest_conn)
        src_conn.close()
        dest_conn.close()

        dest_conn = sqlite3.connect(str(tmp_dest))
        cur = dest_conn.cursor()
        new_artifact_root = (dest_root / "mlflow" / "mlflow_artifacts").as_uri()
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
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=None,
            checksum_source=item.checksum or "",
            checksum_dest=dest_checksum,
            status=MigrationStatus.COPIED,
        )
    except Exception as exc:
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=None,
            checksum_source=item.checksum or "",
            checksum_dest=None,
            status=MigrationStatus.FAILED,
            error=str(exc),
        )


def copy_file(
    src: Path, dest: Path, item: InventoryItem, dry_run: bool
) -> MigrationRecord:
    if dry_run:
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=item.run_id,
            checksum_source=item.checksum or "",
            checksum_dest=None,
            status=MigrationStatus.PENDING,
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and compute_sha256(dest) == item.checksum:
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=item.run_id,
            checksum_source=item.checksum or "",
            checksum_dest=item.checksum,
            status=MigrationStatus.SKIPPED,
        )

    tmp_dest = dest.with_suffix(f"{dest.suffix}.tmp")
    try:
        shutil.copy2(src, tmp_dest)
        tmp_dest.replace(dest)
        dest_checksum = compute_sha256(dest)
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=item.run_id,
            checksum_source=item.checksum or "",
            checksum_dest=dest_checksum,
            status=MigrationStatus.COPIED,
        )
    except Exception as exc:
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=item.run_id,
            checksum_source=item.checksum or "",
            checksum_dest=None,
            status=MigrationStatus.FAILED,
            error=str(exc),
        )


def migrate_tensor(
    src: Path, item: InventoryItem, dest_root: Path, dry_run: bool
) -> MigrationRecord:
    if item.run_id:
        dest = dest_root / "experiments" / item.run_id / src.name
    else:
        dest = dest_root / "unassigned" / "tensors" / src.name

    if dry_run:
        return MigrationRecord(
            source_path=str(src),
            destination_path=str(dest),
            item_type=item.item_type,
            run_id=item.run_id,
            checksum_source=item.checksum or "",
            checksum_dest=None,
            status=MigrationStatus.PENDING,
        )

    return copy_file(src, dest, item, dry_run=False)
