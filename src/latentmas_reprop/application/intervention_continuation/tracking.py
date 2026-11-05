"""Reconcile per-sample MLflow traces with completed local results."""

from pathlib import Path
from typing import Any

from ...domain.models.use_cases import INTERVENTION_CONTINUATION, mlflow_experiment_name
from ...domain.services.prompts import initial_input
from ...infrastructure.paths.resolver import get_git_commit_hash
from ...infrastructure.storage.tensor_store import (
    load_json,
    save_json_atomic,
)
from .artifacts import trace_intervention_continuation_sample
from .identity import canonical_hash, result_identity
from .tracking_verification import (
    fetch_condition_records,
    remove_trace_references,
    search_matching_traces,
    verify_trace_spans,
)

_records = fetch_condition_records
_search = search_matching_traces
_verified = verify_trace_spans
_remove_references = remove_trace_references


def _group(state: Path, args: Any, model: Any) -> dict[str, Any]:
    # The result identity already defines all decoder settings and decoder version.
    probe = result_identity(
        state.name, 0, "qe", {"question": "", "gold": ""}, args, model
    )
    probe.pop("condition")
    probe.pop("gold")
    probe["step"] = int(getattr(args, "latent_steps", 10))
    probe.update(
        {
            "control_signal": getattr(args, "control_signal", "\nSolution:"),
            "receiver_role": getattr(args, "receiver_role", "judger"),
            "prompt": getattr(args, "prompt", "sequential"),
            "think": getattr(args, "think", None),
            "plpe_prompt": getattr(args, "plpe_prompt", None),
        }
    )
    if "initial_states" in load_json(state / "state.json")["identity"]:
        probe["intervention_latent_steps"] = int(
            getattr(args, "intervention_latent_steps", 20)
        )
        probe["sequential"] = result_identity(
            state.name,
            probe["intervention_latent_steps"],
            "latent_intervention",
            {"question": "", "gold": ""},
            args,
            model,
        )
    return probe


def start_continuation_staging_tracker(
    tracker: Any,
    base: Path,
    state: Path,
    args: Any,
    model: Any,
) -> dict[str, Any]:
    """Start an MLflow staging run for real-time per-sample tracing."""
    group = canonical_hash(_group(state, args, model))
    run_dir = state / "runs" / group
    run_dir.mkdir(parents=True, exist_ok=True)
    relative = str(run_dir.relative_to(base))
    config_path = run_dir / "config.json"
    config = {
        "state_path": str(state.relative_to(base)),
        "step": int(getattr(args, "latent_steps", 10)),
        "result_paths": [],
    }
    if config_path.is_file():
        config = load_json(config_path)
        if config.get("state_path") != str(state.relative_to(base)) or config.get(
            "step"
        ) != int(getattr(args, "latent_steps", 10)):
            raise ValueError("Run identity mismatch")
    else:
        save_json_atomic(config_path, config)

    tags = {
        "rel_dir": relative,
        "data_dir": str(run_dir.resolve()),
        "comparison_id": group,
        "experiment_type": INTERVENTION_CONTINUATION,
        "initial_input": "comparison"
        if "state_paths" in config
        else initial_input(args),
    }
    tracker.start_staging_run(
        experiment_name=mlflow_experiment_name(INTERVENTION_CONTINUATION),
        run_name=relative,
        tags=tags,
    )
    tracker.log_params(_group(state, args, model))
    return {
        "group": group,
        "run_dir": run_dir,
        "relative": relative,
        "config_path": config_path,
        "config": config,
        "tags": tags,
    }


def trace_continuation_sample(
    tracker: Any,
    state: Path,
    sample: str,
    index: int,
    item: dict[str, Any],
    args: Any,
    model: Any,
    group: str,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> str | None:
    """Trace a single completed sample in real time during execution."""
    comparison = config is not None and "result_references" in config
    if comparison:
        from .comparison import comparison_meta, comparison_records

        records = comparison_records(state.parents[3], config, sample, item)
    else:
        records = _records(state, sample, item, args, model)
    if not records:
        return None

    if not comparison and config is not None and config_path is not None:
        paths = set(config.get("result_paths", []))
        for record in records.values():
            identity = record["identity"]
            path = (
                Path("samples")
                / sample
                / "results"
                / f"ls{identity['step']}"
                / canonical_hash(identity)
                / identity["condition"]
            )
            paths.add(path.as_posix())
        if paths != set(config["result_paths"]):
            config["result_paths"] = sorted(paths)
            save_json_atomic(config_path, config)

    if comparison:
        meta = comparison_meta(state.parents[3], config, sample, records)
    else:
        meta = load_json(state / "samples" / sample / "shared" / "ls0" / "result.json")
        meta["initial_role"] = next(iter(records.values())).get("initial_role")
        if any(r["identity"]["step"] > 0 for r in records.values()):
            meta["latent_steps"] = int(getattr(args, "latent_steps", 10))

    fingerprint = canonical_hash(records)
    run_id = tracker.active_run_id or ""
    client = getattr(tracker, "client", None)
    git_hash = get_git_commit_hash()

    old: list[Any] = []
    experiment_id = ""
    keep: str | None = None
    if client and run_id:
        experiment_name = mlflow_experiment_name(INTERVENTION_CONTINUATION)
        exp = client.get_experiment_by_name(experiment_name)
        experiment_id = exp.experiment_id if exp else ""
        if experiment_id:
            old = _search(client, run_id, experiment_id, sample, group)
            valid = [
                trace
                for trace in old
                if _verified(trace, sample, group, fingerprint, set(records), run_id)
            ]
            if valid:
                keep = valid[0].info.trace_id

    if not keep:
        trace_id = trace_intervention_continuation_sample(
            tracker,
            args,
            index,
            sample,
            item,
            meta,
            records,
            git_hash,
            comparison_id=group,
            result_fingerprint=fingerprint,
        )
        if not trace_id:
            raise RuntimeError(f"Trace creation failed for sample {sample}")

        if client and run_id:
            trace = client.get_trace(trace_id, flush=True)
            if not _verified(trace, sample, group, fingerprint, set(records), run_id):
                raise RuntimeError(
                    f"Trace verification failed for sample {sample}: {trace_id}"
                )
        keep = trace_id

    if client and experiment_id and old:
        for t in old:
            t_id = t.info.trace_id
            if t_id == keep:
                continue
            try:
                deleted = client.delete_traces(
                    experiment_id=experiment_id, trace_ids=[t_id]
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not delete previous trace {t_id} for sample {sample}: {exc}"
                ) from exc
            if deleted != 1:
                raise RuntimeError(
                    f"Could not delete previous trace {t_id} for sample {sample}"
                )
            _remove_references(state, sample, t_id)
    return keep


def promote_continuation_tracker(
    tracker: Any,
    tracking_ctx: dict[str, Any],
    metrics: dict[str, Any],
    args: Any,
    model: Any,
    state: Path,
) -> Any:
    """Promote the completed staging run to the canonical final run."""
    clean_metrics = {
        key: value for key, value in metrics.items() if isinstance(value, (int, float))
    }
    relative = tracking_ctx["relative"]
    tags = tracking_ctx["tags"]
    params = _group(state, args, model)
    return tracker.promote_staging_run(
        final_run_name=relative,
        tags=tags,
        metrics=clean_metrics,
        params=params,
    )


def track_results(
    tracker: Any,
    base: Path,
    state: Path,
    dataset: list[dict[str, Any]],
    args: Any,
    model: Any,
    metrics: dict[str, Any],
) -> None:
    tracking_ctx = start_continuation_staging_tracker(tracker, base, state, args, model)
    group = tracking_ctx["group"]
    config = tracking_ctx["config"]
    config_path = tracking_ctx["config_path"]
    try:
        for index, item in enumerate(dataset):
            sample = f"{index:06d}"
            trace_continuation_sample(
                tracker=tracker,
                state=state,
                sample=sample,
                index=index,
                item=item,
                args=args,
                model=model,
                group=group,
                config=config,
                config_path=config_path,
            )
        promote_continuation_tracker(
            tracker=tracker,
            tracking_ctx=tracking_ctx,
            metrics=metrics,
            args=args,
            model=model,
            state=state,
        )
    except Exception:
        tracker.discard_staging_run(status="FAILED")
        raise
