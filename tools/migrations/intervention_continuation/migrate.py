"""Convert legacy cumulative intervention states; dry-run unless --apply is set."""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from latentmas_reprop.application.intervention_continuation.identity import (
    canonical_hash,
    state_path,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage.tensor_store import load_json

from .files import publish_json, require_safe_path
from .inventory import discover, group_sources, inspect_samples
from .inventory import samples as source_samples
from .runs import publish_run, publish_run_link
from .samples import publish_sample
from .tracking import find_runs


def migrate(experiments: Path, tracking_uri: str | None, apply: bool) -> dict[str, Any]:
    experiments = experiments.resolve()
    if not experiments.is_dir():
        raise ValueError(f"Experiments directory does not exist: {experiments}")
    links = find_runs(tracking_uri, experiments) if tracking_uri else {}
    sources = discover(experiments)
    run_sources: dict[str, set[str]] = defaultdict(set)
    for source in sources:
        relative = source.relative_to(experiments).as_posix()
        for run_id in links.get(relative, []):
            run_sources[run_id].add(relative)
    ambiguous = {run_id for run_id, refs in run_sources.items() if len(refs) > 1}
    parents: dict[str, list[Path]] = defaultdict(list)
    for source in sources:
        require_safe_path(source, experiments)
        parents[source.parent.relative_to(experiments).as_posix()].append(source)
    groups: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for parent, siblings in parents.items():
        for fingerprint, candidates in group_sources(siblings):
            groups[(parent, fingerprint)].extend(candidates)
    report: dict[str, Any] = {
        "mode": "apply" if apply else "dry-run",
        "sources": [],
        "samples": [],
        "counts": {},
    }
    for (parent, prompt_fingerprint), source_list in groups.items():
        parts = Path(parent).parts
        identity = {
            "task": parts[0],
            "model": parts[1],
            "legacy_group": {
                "parent": parent,
                "prompt_fingerprint": prompt_fingerprint,
            },
            "model_revision": "unknown",
            "tokenizer_revision": "unknown",
        }
        provenance = {"execution_environment": "unknown", "rng_validation": "unknown"}
        state = state_path(experiments, identity)
        require_safe_path(state, experiments)
        if (state / "state.json").is_file() and load_json(state / "state.json") != {
            "identity": identity,
            "provenance": provenance,
        }:
            raise ValueError(f"Existing state identity conflicts: {state}")
        if apply:
            publish_json(
                state / "state.json", {"identity": identity, "provenance": provenance}
            )
        failed_samples: dict[Path, list[str]] = defaultdict(list)
        result_paths: dict[str, list[dict[str, Any]]] = defaultdict(list)
        source_steps: dict[Path, set[int]] = defaultdict(set)
        for source in source_list:
            rel = source.relative_to(experiments).as_posix()
            report["sources"].append(
                {
                    "source": rel,
                    "run_ids": links.get(rel, []),
                    "state_path": state.relative_to(experiments).as_posix(),
                }
            )
        samples = sorted(
            {sample for source in source_list for sample in source_samples(source)}
        )
        inspected = iter(inspect_samples(source_list, samples)) if not apply else None
        for position, sample in enumerate(samples, 1):
            involved = [
                source
                for source in source_list
                if (source / "samples" / sample).is_dir()
            ]
            item: dict[str, Any] = {
                "sample": sample,
                "sources": [p.relative_to(experiments).as_posix() for p in involved],
                "state_path": state.relative_to(experiments).as_posix(),
            }
            try:
                if apply:
                    status, paths = publish_sample(state, sample, involved, experiments)
                    for path in paths:
                        result_paths[path["source"]].append(path)
                else:
                    assert inspected is not None
                    outcome = next(inspected)
                    if isinstance(outcome, Exception):
                        raise outcome
                    _, results, segments = outcome
                    status = "mergeable"
                    item.update(
                        steps=[segment.step for segment in segments],
                        conditions=len(results),
                    )
                for source in involved:
                    meta = load_json(
                        source / "samples" / sample / "shared" / "result.json"
                    )
                    source_steps[source].add(int(meta["latent_steps"]))
                item["status"] = status
            except Exception as exc:
                if isinstance(exc, MemoryError):
                    raise
                for source in involved:
                    failed_samples[source].append(sample)
                item.update(
                    status="conflict" if "Conflicting" in str(exc) else "failed",
                    reason=str(exc),
                )
            report["samples"].append(item)
            if position % 100 == 0:
                print(
                    f"{parent}: {position}/{len(samples)} samples checked",
                    file=sys.stderr,
                    flush=True,
                )
        if apply:
            for source in source_list:
                if not source_steps[source]:
                    continue
                rel = source.relative_to(experiments).as_posix()
                try:
                    if len(source_steps[source]) != 1:
                        raise ValueError(
                            "Source has missing or conflicting latent steps"
                        )
                    run_hash = canonical_hash({"source": rel, "state": state.name})
                    run_dir = state / "runs" / run_hash
                    require_safe_path(run_dir, experiments, allow_published_leaf=True)
                    data = {
                        "source": rel,
                        "run_ids": links.get(rel, []),
                        "config": {
                            "state_path": state.relative_to(experiments).as_posix(),
                            "step": next(iter(source_steps[source])),
                            "result_paths": result_paths[rel],
                            "failed_samples": failed_samples[source],
                        },
                    }
                    publish_run(source, run_dir, data)
                    for run_id in links.get(rel, []):
                        if run_id not in ambiguous:
                            publish_run_link(experiments, run_id, run_dir)
                        else:
                            report["samples"].append(
                                {
                                    "source": rel,
                                    "status": "failed",
                                    "reason": f"Ambiguous MLflow run: {run_id}",
                                }
                            )
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    report["samples"].append(
                        {"source": rel, "status": "failed", "reason": str(exc)}
                    )
    for item in report["samples"]:
        status = item["status"]
        report["counts"][status] = report["counts"].get(status, 0) + 1
        if status in {"mergeable", "migrated", "reused"}:
            key = (
                "multiple_checkpoints"
                if len(item.get("sources", [])) > 1
                else "single_checkpoint"
            )
            report["counts"][key] = report["counts"].get(key, 0) + 1
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments", type=Path, default=PathResolver().experiments_dir
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument(
        "--apply", action="store_true", help="publish validated segments and results"
    )
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    try:
        report = migrate(args.experiments, args.tracking_uri, args.apply)
    except Exception as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return (
        1
        if any(item["status"] in {"failed", "conflict"} for item in report["samples"])
        else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
