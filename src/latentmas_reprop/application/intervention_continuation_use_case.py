"""Application use case for LMAS Intervention Continuation Analysis (PLE, PLCE, PLPE)."""

from typing import Any

import torch
from tqdm import tqdm

from ..domain.ports.cache_port import CachePort
from ..domain.ports.dataset_port import DatasetPort
from ..domain.ports.evaluator_port import EvaluatorPort
from ..domain.ports.tracking_port import ExperimentTrackerPort
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.models.model_wrapper import ModelWrapper
from ..infrastructure.paths.naming import (
    build_experiment_relpath,
    is_overwrite_requested,
    resolve_experiment_dtype_name,
)
from ..infrastructure.paths.resolver import get_git_commit_hash, get_path_resolver
from ..infrastructure.storage import (
    FileStorage,
    get_file_storage,
    save_json_atomic,
)
from .intervention_continuation.artifacts import (
    build_summary_dict,
    log_to_mlflow_tracker,
    save_experiment_summary,
    start_intervention_continuation_tracker,
    trace_intervention_continuation_sample,
)
from .intervention_continuation.checkpoint import (
    find_latent_step_checkpoint,
    make_step_checkpoint_saver,
)
from .intervention_continuation.runner import InterventionContinuationRunner

CONDITIONS = ["ple", "plce", "plpe"]


class InterventionContinuationUseCase:
    """Orchestrates Common Latent calculation and branching across PLE, PLCE, PLPE."""

    def __init__(
        self,
        dataset_port: DatasetPort | None = None,
        cache_port: CachePort | None = None,
        evaluator_port: EvaluatorPort | None = None,
        tracker_port: ExperimentTrackerPort | None = None,
        storage: FileStorage | None = None,
    ) -> None:
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR
        self.tracker_port = tracker_port
        self.storage = storage or get_file_storage()

    @torch.no_grad()
    def execute(
        self,
        model: ModelWrapper,
        args: Any,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        args.intervention_continuation = True
        overwrite = is_overwrite_requested(args)

        dtype_name = getattr(model, "dtype_name", "") or resolve_experiment_dtype_name(args)
        backend = "vllm" if getattr(args, "use_vllm", False) else "transformers"
        exp_relpath = build_experiment_relpath(args, dtype_name=dtype_name, backend=backend)
        args.run_id = exp_relpath
        args.exp_relpath = exp_relpath

        resolver = get_path_resolver()
        exp_dir = resolver.get_experiment_dir(exp_relpath, create=True)
        cfg_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
        save_json_atomic(
            exp_dir / "config.json",
            {
                k: str(v)
                if not isinstance(v, (int, float, bool, list, dict, type(None)))
                else v
                for k, v in cfg_dict.items()
            },
        )

        dataset = list(
            self.dataset_port.load(
                task=getattr(args, "task", "gsm8k"),
                split=getattr(args, "split", "test"),
            )
        )
        max_s = int(getattr(args, "max_samples", 0))
        if max_s > 0:
            dataset = dataset[:max_s]
        n_samples = len(dataset)

        raw_conds = getattr(args, "intervention_conditions", None)
        if raw_conds:
            conditions = (
                [c.strip() for c in raw_conds.split(",") if c.strip()]
                if isinstance(raw_conds, str)
                else list(raw_conds)
            )
        else:
            conditions = ["ple", "plce", "plpe"]

        runner = InterventionContinuationRunner(model, args, self.evaluator_port)
        all_records: list[dict[str, Any]] = []
        git_hash = get_git_commit_hash()
        run_started = start_intervention_continuation_tracker(
            self.tracker_port, exp_dir, exp_relpath, git_hash, args
        )

        show_progress = not bool(getattr(args, "no_progress", False))
        dataset_iter = tqdm(
            dataset,
            desc="Intervention Continuation",
            disable=not show_progress,
            unit="sample",
        )
        for idx, item in enumerate(dataset_iter):
            skey = f"{idx:06d}"
            all_conds_complete = (
                not overwrite
                and self.storage.is_condition_complete(exp_relpath, skey, "shared")
                and all(
                    self.storage.is_condition_complete(exp_relpath, skey, c)
                    for c in conditions
                )
            )
            if all_conds_complete:
                shared_meta = self.storage.load_condition_result(
                    exp_relpath, skey, "shared"
                )
                cond_records: dict[str, dict[str, Any]] = {
                    c: self.storage.load_condition_result(exp_relpath, skey, c)
                    for c in conditions
                }
                if self.tracker_port and any(
                    not r.get("trace_id") for r in cond_records.values()
                ):
                    tid = trace_intervention_continuation_sample(
                        self.tracker_port,
                        args,
                        idx,
                        skey,
                        item,
                        shared_meta,
                        cond_records,
                        git_hash,
                    )
                    if tid:
                        for c, r in cond_records.items():
                            r["trace_id"] = tid
                            self.storage.save_condition(
                                exp_relpath, skey, c, r, tensors=None, overwrite=True
                            )
                all_records.extend(cond_records.values())
                continue

            # UC-01: Shared common latent calculation
            if not overwrite and self.storage.is_condition_complete(
                exp_relpath, skey, "shared"
            ):
                shared_kv = self.storage.load_condition_tensor(
                    exp_relpath, skey, "shared", "kv"
                )
                shared_hidden = self.storage.load_condition_tensor(
                    exp_relpath, skey, "shared", "hidden"
                )
                shared_meta = self.storage.load_condition_result(
                    exp_relpath, skey, "shared"
                )
            else:
                target_ls = int(getattr(args, "latent_steps", 10))
                interval = int(getattr(args, "latent_step_checkpoint_interval", 10))
                init_step, init_kv, init_hidden, init_meta = (
                    (0, None, None, None)
                    if overwrite
                    else find_latent_step_checkpoint(
                        self.storage, args, dtype_name, backend, skey, target_ls, interval
                    )
                )
                ckpt_cb = make_step_checkpoint_saver(
                    self.storage, args, dtype_name, backend, skey, idx, overwrite
                )
                shared_kv, shared_hidden, meta = runner.compute_common_latent(
                    item=dict(item, **(init_meta or {})),
                    initial_kv=init_kv,
                    initial_hiddens=init_hidden,
                    initial_step=init_step,
                    checkpoint_callback=ckpt_cb,
                )
                shared_meta = {
                    "status": "complete",
                    "sample_index": idx,
                    "task": getattr(args, "task", "gsm8k"),
                    "files": ["hidden.pt", "kv.pt"],
                    **meta,
                }
                self.storage.save_condition(
                    exp_relpath,
                    skey,
                    "shared",
                    shared_meta,
                    tensors={"hidden": shared_hidden, "kv": shared_kv},
                    overwrite=overwrite,
                )

            last_hidden = shared_hidden[-1]
            cond_records = {}

            # UC-02: Branch into ple, plce, plpe
            for cond in conditions:
                if not overwrite and self.storage.is_condition_complete(
                    exp_relpath, skey, cond
                ):
                    rec = self.storage.load_condition_result(
                        exp_relpath, skey, cond
                    )
                else:
                    if cond == "ple":
                        branch_h, rec = runner.execute_ple(
                            item, shared_kv, last_hidden
                        )
                    elif cond == "plce":
                        branch_h, rec = runner.execute_plce(item, shared_kv)
                    elif cond == "plpe":
                        branch_h, rec = runner.execute_plpe(item, shared_kv)
                    else:
                        continue

                    rec = {
                        "status": "complete",
                        "sample_index": idx,
                        "files": ["hidden.pt"],
                        **rec,
                    }
                    self.storage.save_condition(
                        exp_relpath,
                        skey,
                        cond,
                        rec,
                        tensors={"hidden": branch_h},
                        overwrite=overwrite,
                    )
                cond_records[cond] = rec

            tid = trace_intervention_continuation_sample(
                self.tracker_port,
                args,
                idx,
                skey,
                item,
                shared_meta,
                cond_records,
                git_hash,
            )
            if tid:
                for c, r in cond_records.items():
                    r["trace_id"] = tid
                    self.storage.save_condition(
                        exp_relpath, skey, c, r, tensors=None, overwrite=True
                    )
            all_records.extend(cond_records.values())

        summary_data = build_summary_dict(
            all_records, conditions, n_samples, cfg_dict
        )
        save_experiment_summary(exp_dir, summary_data, all_records)
        log_to_mlflow_tracker(
            self.tracker_port,
            exp_dir,
            exp_relpath,
            git_hash,
            args,
            summary_data,
            run_already_started=run_started,
        )
        return summary_data, all_records
