"""Helpers for finding and persisting intermediate latent step checkpoints."""

import copy
from typing import Any

import torch

from ...infrastructure.paths.naming import build_experiment_relpath
from ...infrastructure.storage import FileStorage


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
