"""Intervention trace publication for one sample and latent boundary."""

import json
from types import SimpleNamespace

from ..domain.models.use_cases import LATENT_TRAJECTORY, mlflow_experiment_name
from ..infrastructure.paths.resolver import get_git_commit_hash
from ..infrastructure.storage import save_json_atomic
from .intervention_continuation.artifacts import trace_intervention_continuation_sample
from .intervention_continuation.identity import canonical_hash
from .intervention_continuation.tracking_verification import (
    search_matching_traces,
    verify_trace_spans,
)


def reconcile_step_trace(tracker, args, sample_dir, step):
    """Reuse, verify and replace traces with the intervention publication contract."""
    sample = json.loads((sample_dir / "sample.json").read_text())
    rows = [
        json.loads(
            (sample_dir / "continuations" / c / f"step_{step:03d}.json").read_text()
        )
        for c in "abcd"
        if c in args.conditions
    ]
    records = {}
    for row in rows:
        record = dict(row)
        record.pop("trace_id", None)
        # Intervention displays evaluated text and counts non-EOS prediction hiddens.
        record["generated_text"] = row["evaluation_text"]
        record["generated_length"] = row["content_length"]
        records[row["condition"]] = record
    key = f"{sample_dir.name}_step_{step:03d}"
    group = canonical_hash(
        {"directory": str(sample_dir.parent.parent.resolve()), "step": step}
    )
    fingerprint = canonical_hash({"sample": sample, "records": records})
    experiment = tracker.client.get_experiment_by_name(
        mlflow_experiment_name(LATENT_TRAJECTORY)
    )
    old = search_matching_traces(
        tracker.client, tracker.active_run_id, experiment.experiment_id, key, group
    )

    def verified(trace):
        return verify_trace_spans(
            trace,
            key,
            group,
            fingerprint,
            set(records),
            tracker.active_run_id,
            shared_span="initial_states",
        )

    valid = [trace for trace in old if verified(trace)]
    if valid:
        trace_id = valid[0].info.trace_id
        # A retained trace may belong to the preceding canonical run.
        tracker.client.link_traces_to_run(
            trace_ids=[trace_id], run_id=tracker.active_run_id
        )
    else:
        step_args = SimpleNamespace(**vars(args))
        step_args.latent_steps = step
        trace_id = trace_intervention_continuation_sample(
            tracker,
            step_args,
            sample.get("index", int(sample_dir.name)),
            key,
            sample,
            {
                "initial_states": {
                    name: {
                        **meta,
                        "trajectory_latent_steps": meta["latent_steps"],
                        "latent_steps": step,
                        "cache_sequence_length": meta["prompt_length"] + step,
                    }
                    for name, meta in sample["initial_states"].items()
                },
                "initial_input": "rq",
            },
            records,
            get_git_commit_hash(),
            comparison_id=group,
            result_fingerprint=fingerprint,
            trace_tags={
                "latent_step": str(step),
                "dataset_sample_key": sample_dir.name,
            },
        )
        trace = tracker.client.get_trace(trace_id, flush=True)
        if not verified(trace):
            raise RuntimeError(f"Trace verification failed for {key}: {trace_id}")
    for trace in old:
        previous = trace.info.trace_id
        if previous != trace_id and (
            tracker.client.delete_traces(experiment.experiment_id, trace_ids=[previous])
            != 1
        ):
            raise RuntimeError(f"Could not delete previous trace {previous}")
    save_json_atomic(
        sample_dir / "traces" / f"step_{step:03d}.json",
        {
            "trace_id": trace_id,
            "result_fingerprint": fingerprint,
            "comparison_id": group,
        },
    )
    return rows
