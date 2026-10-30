"""Reconcile per-sample MLflow traces with completed local results."""

import json
import tempfile
from pathlib import Path
from typing import Any

from ...domain.models.use_cases import INTERVENTION_CONTINUATION, mlflow_experiment_name
from ...domain.services.prompts import resolve_p1_variant_id
from ...infrastructure.paths.resolver import get_git_commit_hash
from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .artifacts import trace_intervention_continuation_sample
from .identity import canonical_hash, condition_name, result_identity


def _group(state: Path, args: Any, model: Any) -> dict[str, Any]:
    # The result identity already defines all decoder settings and decoder version.
    probe = result_identity(
        state.name, 0, "pe", {"question": "", "gold": ""}, args, model
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
    return probe


def _records(
    state: Path, sample: str, item: dict[str, Any], args: Any, model: Any
) -> dict[str, dict[str, Any]]:
    step = int(getattr(args, "latent_steps", 10))
    variant = resolve_p1_variant_id(args)
    conditions = ["pe", "ple", "plce", "plpe"]
    if variant in {"planner", "critic", "refiner", "judger"}:
        conditions += ["rqlre", "rqlqe"]
        if not getattr(args, "plpe_prompt", None):
            conditions += ["rqrqe"]
    records = {}
    for condition in conditions:
        branch_step = 0 if condition in {"pe", "rqrqe"} else step
        identity = result_identity(
            state.name, branch_step, condition, item, args, model
        )
        directory = (
            state
            / "samples"
            / sample
            / "results"
            / f"ls{branch_step}"
            / canonical_hash(identity)
            / condition
        )
        if not is_condition_complete(directory):
            continue
        record = load_json(directory / "result.json")
        if (
            record.get("status") != "complete"
            or record.get("identity") != identity
            or "hidden.pt" not in record.get("files", [])
        ):
            raise ValueError(f"Invalid completed result: {directory}")
        record.pop("trace_id", None)
        record["condition_name"] = condition_name(condition, args)
        record["initial_role"] = (
            variant if variant in {"planner", "critic", "refiner", "judger"} else None
        )
        if condition in {"plpe", "rqlre", "rqrqe"} and not (
            condition == "plpe" and getattr(args, "plpe_prompt", None)
        ):
            record["receiver_role"] = getattr(args, "receiver_role", "judger")
        records[record["condition_name"]] = record
    return records


def _search(
    client: Any, run_id: str, experiment_id: str, sample: str, group: str
) -> list[Any]:
    traces = []
    token = None
    while True:
        page = client.search_traces(
            run_id=run_id,
            locations=[experiment_id],
            filter_string=f"tags.sample_key = '{sample}' AND tags.comparison_id = '{group}'",
            page_token=token,
            max_results=100,
            include_spans=True,
            flush=True,
        )
        traces.extend(page)
        token = page.token
        if not token:
            return traces


def _verified(
    trace: Any, sample: str, group: str, fingerprint: str, names: set[str], run_id: str
) -> bool:
    tags = trace.info.tags
    roots = [
        span
        for span in trace.data.spans
        if span.name == f"sample_{sample}" and span.parent_id is None
    ]
    if len(roots) != 1:
        return False
    spans = {
        span.name for span in trace.data.spans if span.parent_id == roots[0].span_id
    }
    return (
        trace.info.state == "OK"
        and trace.info.trace_metadata.get("mlflow.sourceRun") == run_id
        and tags.get("sample_key") == sample
        and tags.get("comparison_id") == group
        and tags.get("result_fingerprint") == fingerprint
        and {"common_latent", *(f"decode_{name}" for name in names)} <= spans
    )


def _remove_references(state: Path, sample: str, trace_id: str) -> None:
    for path in (state / "samples" / sample / "results").glob("ls*/*/*/result.json"):
        record = load_json(path)
        if record.get("trace_id") == trace_id:
            record.pop("trace_id")
            save_json_atomic(path, record)
    for path in (state / "runs").glob("*/samples.jsonl"):
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
        changed = False
        for row in rows:
            if (
                row.get("sample_index") == int(sample)
                and row.get("trace_id") == trace_id
            ):
                row.pop("trace_id")
                changed = True
        if changed:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
                try:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)


def track_results(
    tracker: Any,
    base: Path,
    state: Path,
    dataset: list[dict[str, Any]],
    args: Any,
    model: Any,
    metrics: dict[str, Any],
) -> None:
    group = canonical_hash(_group(state, args, model))
    run_dir = state / "runs" / group
    relative = str(run_dir.relative_to(base))
    run = tracker.start_run(
        experiment_name=mlflow_experiment_name(INTERVENTION_CONTINUATION),
        run_name=relative,
        tags={
            "rel_dir": relative,
            "data_dir": str(run_dir.resolve()),
            "comparison_id": group,
            "experiment_type": INTERVENTION_CONTINUATION,
            "p1_variant": resolve_p1_variant_id(args),
        },
    )
    run_id, experiment_id = run.info.run_id, run.info.experiment_id
    tracker.log_params(_group(state, args, model))
    client = tracker.client
    git_hash = get_git_commit_hash()
    try:
        for index, item in enumerate(dataset):
            sample = f"{index:06d}"
            records = _records(state, sample, item, args, model)
            if not records:
                continue
            meta = load_json(
                state / "samples" / sample / "shared" / "ls0" / "result.json"
            )
            meta["initial_role"] = next(iter(records.values())).get("initial_role")
            if any(r["identity"]["step"] > 0 for r in records.values()):
                meta["latent_steps"] = int(getattr(args, "latent_steps", 10))
            fingerprint = canonical_hash(records)
            old = _search(client, run_id, experiment_id, sample, group)
            valid = [
                trace
                for trace in old
                if _verified(trace, sample, group, fingerprint, set(records), run_id)
            ]
            if valid:
                keep = valid[0].info.trace_id
            else:
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
                trace = client.get_trace(trace_id, flush=True)
                if not _verified(
                    trace, sample, group, fingerprint, set(records), run_id
                ):
                    raise RuntimeError(
                        f"Trace verification failed for sample {sample}: {trace_id}"
                    )
                keep = trace_id
            for trace in old:
                old_id = trace.info.trace_id
                if old_id == keep:
                    continue
                try:
                    deleted = client.delete_traces(
                        experiment_id=experiment_id, trace_ids=[old_id]
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not delete previous trace {old_id} for sample {sample}: {exc}"
                    ) from exc
                if deleted != 1:
                    raise RuntimeError(
                        f"Could not delete previous trace {old_id} for sample {sample}"
                    )
                _remove_references(state, sample, old_id)
        tracker.log_metrics(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            }
        )
    except Exception:
        tracker.end_run("FAILED")
        raise
    tracker.end_run("FINISHED")
