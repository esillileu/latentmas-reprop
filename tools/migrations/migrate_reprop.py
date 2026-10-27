"""Migrate legacy reprop benchmark runs into deterministic file-based storage."""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

from latentmas_reprop.application.benchmark.store import BenchmarkRunStore
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver


def parse_legacy_name(name: str):
    """Parse legacy run folder name into parameters."""
    parts = name.split("__")
    if len(parts) < 7:
        return None
    method = parts[0]
    model_name = parts[1].replace("_", "/")
    task = parts[2]
    prompt = parts[3]
    ls = int(parts[4].replace("ls", ""))
    seed = int(parts[5].replace("seed", ""))
    dtype = parts[6]
    rest = parts[7]

    temp_m = re.search(r"temp([\d\.]+)", rest)
    top_m = re.search(r"top([\d\.]+)", rest)
    mt_m = re.search(r"mt(\d+)", rest)
    think_m = re.search(r"think(\d+)", rest)
    realign_m = re.search(r"realign(\d+)", rest)

    args = SimpleNamespace(
        task=task,
        model_name=model_name,
        method=method,
        prompt=prompt,
        latent_steps=ls,
        seed=seed,
        temperature=float(temp_m.group(1)) if temp_m else 0.6,
        top_p=float(top_m.group(1)) if top_m else 0.95,
        max_new_tokens=int(mt_m.group(1)) if mt_m else 2048,
        think=bool(int(think_m.group(1))) if think_m else False,
        latent_space_realign=bool(int(realign_m.group(1))) if realign_m else False,
        split="test",
    )
    return args, dtype


def migrate_reprop_runs(
    source_root: Path,
    dest_root: Path,
    dry_run: bool = False,
) -> list[dict]:
    runs_dir = source_root / "evaluation" / "runs"
    if not runs_dir.is_dir():
        print(f"[Error] Source directory {runs_dir} not found.", file=sys.stderr)
        return []

    dest_experiments = dest_root / "experiments"
    resolver = PathResolver(dest_root.parent)

    migrated_runs = []

    candidates = sorted(
        p for p in runs_dir.iterdir() if p.is_dir() and not p.is_symlink()
    )

    for p in candidates:
        if not any(k in p.name for k in ["baseline__", "text_mas__", "latent_mas__"]):
            continue

        smp = p / "samples.jsonl"
        sm = p / "summary.json"
        if not (smp.is_file() and sm.is_file()):
            continue

        try:
            summary = json.loads(sm.read_text(encoding="utf-8"))
            count = summary.get("total_samples") or summary.get("sample_count") or 0
        except Exception:
            continue

        if count < 1319:
            # Skip partial/smoke runs
            continue

        parsed = parse_legacy_name(p.name)
        if not parsed:
            continue

        args, dtype = parsed
        rel_path = build_experiment_relpath(args, dtype_name=dtype)
        target_dir = dest_experiments / rel_path

        print(f"[*] Migrating: {p.name}")
        print(f"    -> {rel_path} ({count} samples)")

        if dry_run:
            migrated_runs.append({
                "source": str(p),
                "rel_path": rel_path,
                "target": str(target_dir),
                "samples": count,
                "status": "dry_run",
            })
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # 1. Copy config and summary
        target_sm = target_dir / "summary.json"
        target_sm.write_text(sm.read_text(encoding="utf-8"), encoding="utf-8")

        cfg = p / "config.json"
        target_cfg = target_dir / "config.json"
        if cfg.is_file():
            target_cfg.write_text(cfg.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Generate config from summary
            cfg_data = summary.get("config", {})
            target_cfg.write_text(
                json.dumps(cfg_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )

        # 2. Copy samples.jsonl
        target_smp = target_dir / "samples.jsonl"
        target_smp.write_text(smp.read_text(encoding="utf-8"), encoding="utf-8")

        # 3. Partition samples into condition dirs
        samples_dir = target_dir / "samples"
        samples_written = 0
        with open(smp, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                idx = int(record.get("sample_index", line_idx))
                idx_str = f"{idx:06d}"
                cond_dir = samples_dir / idx_str / "benchmark"
                cond_dir.mkdir(parents=True, exist_ok=True)
                res_file = cond_dir / "result.json"
                if not res_file.exists():
                    tmp_file = res_file.with_suffix(".tmp")
                    tmp_file.write_text(
                        json.dumps(record, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    tmp_file.replace(res_file)
                samples_written += 1

        # 4. Verify with BenchmarkRunStore
        store = BenchmarkRunStore(run_id=rel_path, path_resolver=resolver)
        is_ok = store.is_complete(target_samples=1319)
        loaded_count = len(store.read_records())

        if not is_ok or loaded_count != 1319:
            print(
                f"[Warning] Store verification mismatch for {rel_path}: {loaded_count}/1319",
                file=sys.stderr,
            )

        migrated_runs.append({
            "source": str(p),
            "rel_path": rel_path,
            "target": str(target_dir),
            "samples_written": samples_written,
            "verified": is_ok,
            "loaded_count": loaded_count,
            "status": "migrated",
        })

    return migrated_runs


if __name__ == "__main__":
    src = Path(".cache")
    dst = Path("storage")
    dry = "--dry-run" in sys.argv

    results = migrate_reprop_runs(src, dst, dry_run=dry)
    print(f"\n[Done] Total runs processed: {len(results)}")
    all_ok = all(r.get("verified", False) for r in results) if not dry else True
    print(f"All runs fully verified (1319 samples): {all_ok}")
