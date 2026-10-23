"""Verify the integrity and completeness of migrated v2 storage against legacy v1 source."""

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

from .schemas import VerificationReport


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


class MigrationVerifier:
    """Validates that all legacy data was faithfully migrated without corruption or loss."""

    def __init__(self, source_root: Path, dest_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.dest_root = dest_root.resolve()
        self.report = VerificationReport(
            source_root=str(self.source_root),
            destination_root=str(self.dest_root),
        )

    def verify(self) -> VerificationReport:
        print(f"[*] Verifying migrated destination at: {self.dest_root}")

        # 1. Verify SQLite DB
        self._verify_sqlite_db()

        # 2. Verify migration manifest records if present
        manifest_file = self.dest_root / "migration_manifest.json"
        if not manifest_file.is_file():
            self.report.errors.append(
                "Missing migration_manifest.json in destination root."
            )
            print(
                "[Error] Missing migration_manifest.json. Was migration executed?",
                file=sys.stderr,
            )
            return self.report

        try:
            data = json.loads(manifest_file.read_text(encoding="utf-8"))
            records = data.get("records", [])
        except Exception as exc:
            self.report.errors.append(f"Failed to parse migration_manifest.json: {exc}")
            return self.report

        for r in records:
            self.report.total_checked += 1
            dest_path = Path(r["destination_path"])
            expected_hash = r.get("checksum_source")

            if not dest_path.is_file():
                self.report.missing_count += 1
                self.report.errors.append(f"Missing file in destination: {dest_path}")
                continue

            # SQLite DB is modified to update artifact URIs, so skip raw hash check on DB itself
            if dest_path.name == "mlflow.db":
                self.report.success_count += 1
                continue

            actual_hash = compute_sha256(dest_path)
            if expected_hash and actual_hash != expected_hash:
                self.report.mismatch_count += 1
                self.report.errors.append(
                    f"Checksum mismatch for {dest_path}: expected {expected_hash}, found {actual_hash}"
                )
            else:
                self.report.success_count += 1

        # 3. Verify Tensor Manifests
        self._verify_tensor_manifests()

        return self.report

    def _verify_sqlite_db(self) -> None:
        db_path = self.dest_root / "mlflow" / "mlflow.db"
        if not db_path.is_file():
            self.report.errors.append(f"SQLite DB missing at {db_path}")
            self.report.db_valid = False
            return

        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM runs")
            count = cursor.fetchone()[0]
            conn.close()
            self.report.db_valid = True
            print(f"[*] SQLite DB verified: {count} runs readable.")
        except Exception as exc:
            self.report.errors.append(f"SQLite DB check failed: {exc}")
            self.report.db_valid = False

    def _verify_tensor_manifests(self) -> None:
        tensors_root = self.dest_root / "tensors"
        if not tensors_root.is_dir():
            return

        for manifest_path in tensors_root.rglob("tensor_manifest.json"):
            try:
                manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
                entries = manifest_dict.get("entries", {})
                for key, entry in entries.items():
                    uri = entry.get("storage_uri", "")
                    if uri.startswith("file://"):
                        tensor_file = Path(uri[7:])
                    else:
                        tensor_file = Path(uri)
                    if not tensor_file.is_file():
                        self.report.missing_count += 1
                        self.report.errors.append(
                            f"Tensor file missing: {tensor_file} for {key}"
                        )
                    else:
                        actual_checksum = compute_sha256(tensor_file)
                        expected_checksum = entry.get("checksum")
                        if actual_checksum != expected_checksum:
                            self.report.mismatch_count += 1
                            self.report.errors.append(
                                f"Tensor checksum mismatch for {tensor_file}: expected {expected_checksum}, found {actual_checksum}"
                            )
            except Exception as exc:
                self.report.errors.append(
                    f"Failed to verify tensor manifest at {manifest_path}: {exc}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify integrity of migrated v2 storage against legacy v1 source."
    )
    parser.add_argument(
        "--source", required=True, help="Path to legacy storage root (e.g. .cache)"
    )
    parser.add_argument(
        "--destination", required=True, help="Path to new storage root (e.g. storage)"
    )
    parser.add_argument(
        "--report", help="Optional path to output verification report JSON"
    )
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.destination)

    verifier = MigrationVerifier(source_root=source, dest_root=dest)
    report = verifier.verify()

    print("\n=== Migration Verification Summary ===")
    print(f"Destination Root: {report.destination_root}")
    print(f"Total Checked:    {report.total_checked}")
    print(f"Verified OK:      {report.success_count}")
    print(f"Mismatches:       {report.mismatch_count}")
    print(f"Missing Files:    {report.missing_count}")
    print(f"Database Valid:   {report.db_valid}")
    print(f"Overall Status:   {'PASSED' if report.is_valid else 'FAILED'}")
    print("======================================\n")

    if report.errors:
        print("[!] Detected Issues:")
        for err in report.errors[:10]:
            print(f"  - {err}")
        if len(report.errors) > 10:
            print(f"  ... and {len(report.errors) - 10} more errors.")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Verification report saved to: {report_path}")

    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
