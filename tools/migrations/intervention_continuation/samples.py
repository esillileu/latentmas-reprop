"""Stage and verify sample states and legacy results before publication."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from latentmas_reprop.application.intervention_continuation.identity import (
    canonical_hash,
)
from latentmas_reprop.infrastructure.storage.tensor_store import (
    FileStorage,
    is_condition_complete,
    load_json,
    load_tensor,
)

from .checkpoints import Checkpoint, equal_kv, split_checkpoints
from .files import file_hash, manifest, rename_new, require_safe_path
from .inventory import CONDITIONS


def load_sample(
    sources: list[Path],
    sample: str,
):
    checkpoints = []
    results = []
    expected = None
    for source in sources:
        root = source / "samples" / sample
        if not root.is_dir():
            continue
        shared = root / "shared"
        if not is_condition_complete(shared):
            raise ValueError(f"Incomplete shared checkpoint: {shared}")
        meta = load_json(shared / "result.json")
        if meta.get("status") != "complete" or not {"kv.pt", "hidden.pt"}.issubset(
            meta.get("files", [])
        ):
            raise ValueError(f"Invalid shared checkpoint: {shared}")
        sample_identity = meta.get("sample_index")
        if (
            not sample.isdigit()
            or len(sample) != 6
            or sample_identity != int(sample)
            or (expected is not None and sample_identity != expected)
        ):
            raise ValueError(f"Missing or conflicting sample identity: {shared}")
        expected = sample_identity
        step = int(meta["latent_steps"])
        kv = load_tensor(shared / "kv.pt", map_location="cpu")
        hidden = load_tensor(shared / "hidden.pt", map_location="cpu")
        checkpoints.append(Checkpoint(step, kv, hidden, meta))
        for condition in CONDITIONS:
            directory = root / condition
            if not directory.exists():
                continue
            if not is_condition_complete(directory):
                raise ValueError(f"Incomplete condition: {directory}")
            record = load_json(directory / "result.json")
            if (
                record.get("status") != "complete"
                or record.get("condition") != condition
                or record.get("sample_index") != int(sample)
                or not isinstance(record.get("files"), list)
                or "hidden.pt" not in record["files"]
                or any(
                    not isinstance(name, str)
                    or not name
                    or Path(name).name != name
                    or name == "result.json"
                    for name in record["files"]
                )
                or "identity" in record
            ):
                raise ValueError(f"Invalid legacy condition: {directory}")
            tensor = load_tensor(directory / "hidden.pt", map_location="cpu")
            if not torch.is_tensor(tensor):
                raise ValueError(f"Invalid condition tensor: {directory}")
            results.append((source, condition, step, record, tensor))
    return checkpoints, results, split_checkpoints(checkpoints)


def _same_tree(left: Path, right: Path) -> bool:
    if any(path.is_symlink() for root in (left, right) for path in root.rglob("*")):
        raise ValueError("Symbolic link in sample destination")
    files_a = {p.relative_to(left) for p in left.rglob("*") if p.is_file()}
    files_b = {p.relative_to(right) for p in right.rglob("*") if p.is_file()}
    if files_a != files_b:
        return False
    for rel in files_a:
        a, b = left / rel, right / rel
        tensor_file = (
            rel.parts[0] == "shared" and rel.name in {"kv.pt", "hidden.pt"}
        ) or (rel.parts[0] == "results" and rel.name == "hidden.pt")
        if not tensor_file:
            if file_hash(a) != file_hash(b):
                return False
            continue
        x, y = load_tensor(a, "cpu"), load_tensor(b, "cpu")
        if torch.is_tensor(x) and torch.is_tensor(y):
            if x.dtype != y.dtype or not torch.equal(x, y):
                return False
        elif not equal_kv(x, y):
            return False
    return True


def _archive_sample(source: Path, destination: Path) -> None:
    ignored = {"shared/kv.pt", "shared/hidden.pt"}

    def source_manifest() -> dict[str, str]:
        result = {}
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Symbolic link in source: {path}")
            rel = path.relative_to(source).as_posix()
            if path.is_file() and rel not in ignored:
                result[rel] = file_hash(path)
            elif not path.is_file() and not path.is_dir():
                raise ValueError(f"Unsupported source file: {path}")
        return result

    before = source_manifest()

    def ignore(path: str, names: list[str]) -> set[str]:
        return (
            {"kv.pt", "hidden.pt"}.intersection(names)
            if Path(path) == source / "shared"
            else set()
        )

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)
    if manifest(destination) != before or source_manifest() != before:
        raise ValueError("Source sample changed during copy")


def publish_sample(
    state: Path,
    sample: str,
    sources: list[Path],
    experiments: Path,
) -> tuple[str, list[dict[str, Any]]]:
    checkpoints, results, segments = load_sample(sources, sample)
    require_safe_path(
        state / "samples" / sample, experiments, allow_published_leaf=True
    )
    state.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".sample-", dir=state))
    store = FileStorage()
    paths = []
    try:
        for segment in segments:
            store.save_shared_segment(
                temporary,
                sample,
                segment.metadata["start_step"],
                segment.step,
                segment.kv,
                segment.hidden,
                segment.metadata,
            )
        for checkpoint in checkpoints:
            kv, hidden, _ = store.assemble_shared_state(
                temporary, sample, checkpoint.step
            )
            if not equal_kv(kv, checkpoint.kv) or not torch.equal(
                hidden, checkpoint.hidden
            ):
                raise ValueError("New loader differs from cumulative checkpoint")
        for source, condition, step, record, hidden in results:
            rel_source = source.relative_to(experiments).as_posix()
            original = source / "samples" / sample / condition
            reference = {
                "kind": "legacy_intervention_result",
                "source": rel_source,
                "sample": sample,
                "step": 0 if condition == "pe" else step,
                "condition": condition,
                "original_result_sha256": file_hash(original / "result.json"),
            }
            result_hash = canonical_hash(reference)
            rel = (
                Path("samples")
                / sample
                / "results"
                / f"ls{reference['step']}"
                / result_hash
                / condition
            )
            target = temporary / rel
            target.mkdir(parents=True)
            for name in record["files"]:
                if name != "hidden.pt":
                    shutil.copy2(original / name, target / name)
                    if file_hash(original / name) != file_hash(target / name):
                        raise ValueError("Condition file copy failed verification")
            store.save_intervention_result(target, reference, record, hidden)
            loaded, saved = store.load_intervention_result(temporary / rel, reference)
            if not torch.equal(loaded, hidden) or any(
                saved.get(k) != v for k, v in record.items()
            ):
                raise ValueError("New result differs from original")
            paths.append(
                {
                    "source": rel_source,
                    "path": rel.as_posix(),
                    "condition": condition,
                    "trace_id": record.get("trace_id"),
                }
            )
        for source in sources:
            rel_source = source.relative_to(experiments).as_posix()
            archive = (
                temporary
                / "samples"
                / sample
                / "legacy_results"
                / canonical_hash(rel_source)
            )
            _archive_sample(source / "samples" / sample, archive)
        current, current_results, _ = load_sample(sources, sample)
        if len(current) != len(checkpoints) or any(
            a.metadata != b.metadata
            or not equal_kv(a.kv, b.kv)
            or a.hidden.dtype != b.hidden.dtype
            or not torch.equal(a.hidden, b.hidden)
            for a, b in zip(checkpoints, current, strict=True)
        ):
            raise ValueError("Source checkpoint changed during migration")
        if len(current_results) != len(results) or any(
            a[:4] != b[:4] or a[4].dtype != b[4].dtype or not torch.equal(a[4], b[4])
            for a, b in zip(results, current_results, strict=True)
        ):
            raise ValueError("Source condition changed during migration")
        staged = temporary / "samples" / sample
        final = state / "samples" / sample
        final.parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            if not _same_tree(staged, final):
                raise ValueError(f"Existing sample conflicts: {final}")
            return "reused", paths
        rename_new(staged, final, same=_same_tree)
        return "migrated", paths
    finally:
        shutil.rmtree(temporary)
