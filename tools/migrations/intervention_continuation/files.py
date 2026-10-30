"""Immutable, verified publication of migration directories."""

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def require_safe_path(
    path: Path, root: Path, allow_published_leaf: bool = False
) -> None:
    """Reject traversal and symlinked path components below the storage root."""
    if not path.is_relative_to(root):
        raise ValueError(f"Path outside experiments directory: {path}")
    current = root
    for part in path.relative_to(root).parts:
        if part in {".", ".."}:
            raise ValueError(f"Invalid path component: {path}")
        current = current / part
        if current.is_symlink():
            if not allow_published_leaf or current != path:
                raise ValueError(f"Symbolic link in migration path: {current}")
            data_root = path.parent.parent / ".migration-data" / path.parent.name
            if data_root.is_symlink() or data_root.parent.is_symlink():
                raise ValueError(f"Symbolic link in migration data: {data_root}")
            if (current.parent / os.readlink(current)).is_symlink():
                raise ValueError(
                    f"Symbolic link in published migration data: {current}"
                )
            target = current.resolve(strict=True)
            if (
                target.parent != data_root.resolve()
                or not target.name.startswith(f".{path.name}-")
                or not target.is_dir()
            ):
                raise ValueError(f"Invalid published migration path: {current}")


def rename_new(source: Path, destination: Path, same=None) -> None:
    """Publish verified data with an exclusive atomic symlink as completion marker."""
    data_root = destination.parent.parent / ".migration-data" / destination.parent.name
    if data_root.is_symlink() or data_root.parent.is_symlink():
        raise ValueError(f"Symbolic link in migration data: {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    lock = data_root / f".{destination.name}.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(str(destination))
        candidates = sorted(data_root.glob(f".{destination.name}-*"))
        for candidate in candidates:
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError(f"Invalid unpublished data: {candidate}")
            if not any(candidate.iterdir()):
                continue
            matches = (
                same(source, candidate)
                if same
                else manifest(source) == manifest(candidate)
            )
            if not matches:
                raise ValueError(f"Unpublished data conflicts: {candidate}")
        populated = [candidate for candidate in candidates if any(candidate.iterdir())]
        if populated:
            data = populated[0]
        else:
            data = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=data_root))
            try:
                os.rename(source, data)
            except BaseException:
                data.rmdir()
                raise
        # Creating the link is exclusive and makes the verified tree visible at once.
        os.symlink(
            os.path.relpath(data, destination.parent),
            destination,
            target_is_directory=True,
        )


def publish_json(destination: Path, data: Any) -> None:
    """Create an immutable JSON file, accepting only identical existing data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".json-", dir=destination.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, sort_keys=True, ensure_ascii=False)
        try:
            os.link(temp, destination)
        except FileExistsError:
            if (
                destination.is_symlink()
                or not destination.is_file()
                or json.loads(destination.read_text()) != data
            ):
                raise ValueError(
                    f"Existing metadata conflicts: {destination}"
                ) from None
    finally:
        temp.unlink(missing_ok=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symbolic link in source: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = file_hash(path)
        elif not path.is_dir():
            raise ValueError(f"Unsupported source file: {path}")
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
