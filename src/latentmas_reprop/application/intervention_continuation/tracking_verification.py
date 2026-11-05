"""Verification and record gathering helpers for intervention continuation tracking."""

import json
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...domain.services.prompts import initial_role
from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    load_json,
    save_json_atomic,
)
from .conditions import COMPARISON_CONDITIONS, SEQUENTIAL_CONDITIONS, SEQUENTIAL_ROLES
from .identity import canonical_hash, condition_name, result_identity


def fetch_condition_records(
    state: Path, sample: str, item: dict[str, Any], args: Any, model: Any
) -> dict[str, dict[str, Any]]:
    """Gather completed condition records for a sample."""
    step = int(getattr(args, "latent_steps", 10))
    from ...domain.services.prompts.intervention import initial_conditions

    conditions = initial_conditions(args)
    records = {}
    for condition in conditions:
        branch_step = 0 if condition in {"qe", "rqe", "rqrqe"} else step
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
        record["initial_role"] = initial_role(args)
        if condition in {"rqlrqe", "rqlre", "rqrqe"} and not (
            condition == "rqlrqe" and getattr(args, "plpe_prompt", None)
        ):
            record["receiver_role"] = getattr(args, "receiver_role", "judger")
        records[record["condition_name"]] = record
    return records


def search_matching_traces(
    client: Any, run_id: str, experiment_id: str, sample: str, group: str
) -> list[Any]:
    """Search for traces matching sample key and group within an experiment and run."""
    traces = []
    token = None
    filter_str = f"tags.sample_key = '{sample}' AND tags.comparison_id = '{group}'"
    if run_id:
        with suppress(Exception):
            while True:
                page = client.search_traces(
                    run_id=run_id,
                    locations=[experiment_id],
                    filter_string=filter_str,
                    page_token=token,
                    max_results=100,
                    include_spans=True,
                    flush=True,
                )
                traces.extend(page)
                token = page.token
                if not token:
                    break
    if not traces:
        token = None
        with suppress(Exception):
            while True:
                page = client.search_traces(
                    locations=[experiment_id],
                    filter_string=filter_str,
                    page_token=token,
                    max_results=100,
                    include_spans=True,
                    flush=True,
                )
                traces.extend(page)
                token = page.token
                if not token:
                    break
    return traces


def verify_trace_spans(
    trace: Any,
    sample: str,
    group: str,
    fingerprint: str,
    names: set[str],
    run_id: str = "",
    shared_span: str | None = None,
) -> bool:
    """Verify trace integrity, spans, tags, and source run."""
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
    source_run = trace.info.trace_metadata.get("mlflow.sourceRun")
    shared_span = shared_span or (
        "initial_states"
        if names & {name.upper() for name in COMPARISON_CONDITIONS}
        else "common_latent"
    )
    for name in names & {condition.upper() for condition in SEQUENTIAL_CONDITIONS}:
        branch = [
            span
            for span in trace.data.spans
            if span.name == f"decode_{name}" and span.parent_id == roots[0].span_id
        ]
        if len(branch) != 1 or not set(SEQUENTIAL_ROLES) <= {
            span.name
            for span in trace.data.spans
            if span.parent_id == branch[0].span_id
        }:
            return False
    run_matches = True
    if run_id:
        run_matches = (source_run == run_id) or (tags.get("comparison_id") == group)
    return (
        trace.info.state == "OK"
        and run_matches
        and tags.get("sample_key") == sample
        and tags.get("comparison_id") == group
        and tags.get("result_fingerprint") == fingerprint
        and {shared_span, *(f"decode_{name}" for name in names)} <= spans
    )


def remove_trace_references(state: Path, sample: str, trace_id: str) -> None:
    """Strip stale trace_id references from disk result records and summaries."""
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
