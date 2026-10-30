"""Discover legacy intervention continuation directories and samples."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from latentmas_reprop.application.intervention_continuation.identity import (
    canonical_hash,
)
from latentmas_reprop.domain.services.kv_cache import _dense_layers
from latentmas_reprop.infrastructure.storage.tensor_store import load_json, load_tensor

CONDITIONS = ("pe", "ple", "plce", "plpe")


def samples(source: Path) -> list[str]:
    result = []
    for path in (source / "samples").iterdir():
        if path.is_symlink():
            raise ValueError(f"Symbolic link in sample directory: {path}")
        if path.is_dir():
            result.append(path.name)
    return sorted(result)


def discover(experiments: Path) -> list[Path]:
    return sorted(
        {
            path
            for pattern in ("*intervention_continuation*", "*single_intervention*")
            for path in experiments.glob(f"*/*/*/*/{pattern}")
            if path.is_dir() and (path / "samples").is_dir()
        }
    )


def group_sources(siblings: list[Path]) -> list[tuple[str, list[Path]]]:
    """Select candidates using recorded prompt tokens or identical prefill tensors."""
    common = set.intersection(*(set(samples(source)) for source in siblings))
    if not common:
        return [("unknown", siblings)]
    sample = min(common)
    records = []
    try:
        for source in siblings:
            root = source / "samples" / sample / "shared"
            meta = load_json(root / "result.json")
            tokens = meta.get("prompt_token_ids")
            if not tokens:
                return [("unknown", siblings)]
            records.append(
                (source, canonical_hash(tokens), int(meta["prompt_length"]), root)
            )
    except (OSError, ValueError, TypeError, KeyError):
        return [("unknown", siblings)]
    parent = list(range(len(records)))
    prefixes = {}

    def prefix(index: int):
        if index not in prefixes:
            _, _, length, root = records[index]
            layers, _ = _dense_layers(load_tensor(root / "kv.pt", "cpu"))
            hidden = load_tensor(root / "hidden.pt", "cpu")
            prefixes[index] = (
                [(k[..., :length, :], v[..., :length, :]) for k, v in layers],
                hidden[:length],
            )
        return prefixes[index]

    def find(index: int) -> int:
        while parent[index] != index:
            index = parent[index]
        return index

    try:
        for index, (_, fingerprint, length, _) in enumerate(records):
            for other in range(index):
                same = fingerprint == records[other][1]
                if not same and length == records[other][2]:
                    a, ah = prefix(index)
                    b, bh = prefix(other)
                    same = (
                        len(a) == len(b)
                        and ah.dtype == bh.dtype
                        and torch.equal(ah, bh)
                        and all(
                            x.dtype == y.dtype and torch.equal(x, y)
                            for la, lb in zip(a, b, strict=True)
                            for x, y in zip(la, lb, strict=True)
                        )
                    )
                if same:
                    parent[find(index)] = find(other)
    except Exception as exc:
        if isinstance(exc, MemoryError):
            raise
        return [("unknown", siblings)]
    grouped = {}
    for index, (source, fingerprint, _, _) in enumerate(records):
        grouped.setdefault(find(index), []).append((source, fingerprint))
    return [
        (min(fp for _, fp in group), [source for source, _ in group])
        for group in grouped.values()
    ]


def inspect_samples(sources: list[Path], keys: list[str]):
    """Bound parallel CPU validation to four independently loaded samples."""
    from .samples import load_sample

    def inspect(sample: str):
        involved = [p for p in sources if (p / "samples" / sample).is_dir()]
        try:
            return load_sample(involved, sample)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=4) as executor:
        for offset in range(0, len(keys), 4):
            yield from executor.map(inspect, keys[offset : offset + 4])
