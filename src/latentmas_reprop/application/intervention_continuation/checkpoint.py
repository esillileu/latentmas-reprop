"""Helpers for finding and persisting intermediate latent step checkpoints."""

import copy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from ...infrastructure.paths.naming import build_experiment_relpath
from ...infrastructure.paths.resolver import get_git_commit_hash
from ...infrastructure.storage import FileStorage, get_file_storage
from .artifacts import (
    log_to_mlflow_tracker,
    start_intervention_continuation_tracker,
    trace_intervention_continuation_sample,
)


def find_latent_step_checkpoint(
    storage: FileStorage,
    args: Any,
    dtype_name: str,
    backend: str,
    skey: str,
    target_steps: int,
    interval: int,
) -> tuple[int, Any, torch.Tensor | None, dict[str, Any] | None]:
    """Search for the highest available intermediate latent checkpoint step < target_steps."""
    if interval <= 0 or target_steps <= 0:
        return 0, None, None, None

    max_candidate = target_steps - 1
    aligned_max = max_candidate - (max_candidate % interval)
    if aligned_max <= 0:
        return 0, None, None, None

    for step in range(aligned_max, 0, -interval):
        step_args = copy.copy(args)
        step_args.latent_steps = step
        relpath = build_experiment_relpath(
            step_args, dtype_name=dtype_name, backend=backend
        )
        if storage.is_condition_complete(relpath, skey, "shared"):
            try:
                kv = storage.load_condition_tensor(relpath, skey, "shared", "kv")
                hidden = storage.load_condition_tensor(
                    relpath, skey, "shared", "hidden"
                )
                meta = storage.load_condition_result(relpath, skey, "shared")
                return step, kv, hidden, meta
            except Exception:
                continue

    return 0, None, None, None


def make_step_checkpoint_saver(
    storage: FileStorage,
    args: Any,
    dtype_name: str,
    backend: str,
    skey: str,
    idx: int,
    overwrite: bool,
):
    """Create a callback that persists intermediate latent checkpoints."""

    def callback(step: int, kv: Any, hidden: torch.Tensor, meta: dict[str, Any]):
        step_args = copy.copy(args)
        step_args.latent_steps = step
        relpath = build_experiment_relpath(
            step_args, dtype_name=dtype_name, backend=backend
        )
        payload = {
            "status": "complete",
            "sample_index": idx,
            "task": getattr(args, "task", "gsm8k"),
            "files": ["hidden.pt", "kv.pt"],
            **meta,
        }
        storage.save_condition(
            relpath,
            skey,
            "shared",
            payload,
            tensors={"hidden": hidden, "kv": kv},
            overwrite=overwrite,
        )

    return callback


def backfill_intervention_continuation_traces(
    tracker: Any,
    exp_dir: Path,
    exp_relpath: str,
    args: Any,
    summary_data: dict[str, Any],
    conditions: list[str],
    target_count: int,
    storage: FileStorage | None = None,
) -> None:
    """Backfill MLflow sample traces from completed local result.json files and log summary."""
    if not tracker:
        return
    store = storage or get_file_storage()
    git_hash = get_git_commit_hash()
    run_started = start_intervention_continuation_tracker(
        tracker, exp_dir, exp_relpath, git_hash, args
    )
    show_progress = not bool(getattr(args, "no_progress", False))
    for idx in tqdm(
        range(target_count),
        desc="Syncing MLflow traces",
        disable=not show_progress,
        unit="sample",
    ):
        skey = f"{idx:06d}"
        cond_records = {
            c: store.load_condition_result(exp_relpath, skey, c)
            for c in conditions
            if store.is_condition_complete(exp_relpath, skey, c)
        }
        if not cond_records or all(r.get("trace_id") for r in cond_records.values()):
            continue
        shared_meta = (
            store.load_condition_result(exp_relpath, skey, "shared")
            if store.is_condition_complete(exp_relpath, skey, "shared")
            else {}
        )
        tid = trace_intervention_continuation_sample(
            tracker,
            args,
            idx,
            skey,
            {},
            shared_meta,
            cond_records,
            git_hash,
            flush=((idx + 1) % 25 == 0),
        )
        if tid:
            for c, r in cond_records.items():
                r["trace_id"] = tid
                store.save_condition(
                    exp_relpath, skey, c, r, tensors=None, overwrite=True
                )
    log_to_mlflow_tracker(
        tracker,
        exp_dir,
        exp_relpath,
        git_hash,
        args,
        summary_data,
        run_already_started=run_started,
    )
