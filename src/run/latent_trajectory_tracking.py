"""Publish locally saved latent trajectory results using the shared MLflow tracker."""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from threading import Event

from mlflow.entities import Metric

from latentmas_reprop.application.intervention_continuation.artifacts import (
    build_summary_dict,
    save_experiment_summary,
)
from latentmas_reprop.application.latent_trajectory_readout import is_readout_complete
from latentmas_reprop.application.latent_trajectory_tracking import reconcile_step_trace
from latentmas_reprop.domain.models.use_cases import (
    LATENT_TRAJECTORY,
    mlflow_experiment_name,
)
from latentmas_reprop.infrastructure.storage import save_json_atomic
from latentmas_reprop.infrastructure.tracking import MLflowTracker


def publish_step(tracker, args, sample_dir, step):
    """Upload one completed sample/step with its requested condition spans."""
    rows = reconcile_step_trace(tracker, args, sample_dir, step)
    history = []
    for row in rows:
        for name, value in {
            f"samples/{sample_dir.name}/{row['condition']}/correct": int(
                row["correct"]
            ),
            f"samples/{sample_dir.name}/{row['condition']}/generated_length": row[
                "content_length"
            ],
        }.items():
            tracker.client.log_metric(tracker.active_run_id, name, value, step=step)
            history.append(Metric(name, float(value), int(time.time() * 1000), step))
    return {
        "completed_continuations": len(rows),
        "correct_continuations": sum(bool(row["correct"]) for row in rows),
        "history": history,
    }


@contextmanager
def tracked_sample(sample_dir, *, directory, args, config, upload_disabled=None):
    """Keep inference running after publication fails, until the next invocation."""
    upload_disabled = upload_disabled if upload_disabled is not None else Event()
    if upload_disabled.is_set():
        yield lambda step: None
        return
    entered = False
    failure = None

    def disable(exc):
        upload_disabled.set()
        logging.getLogger(__name__).warning(
            "MLflow upload stopped; local computation will continue: %s", exc
        )
        save_json_atomic(
            sample_dir / "upload_failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )

    try:
        with _tracked_sample(
            sample_dir,
            directory=directory,
            args=args,
            config=config,
            upload_disabled=upload_disabled,
            on_failure=disable,
        ) as submit:
            entered = True
            try:
                yield submit
            except BaseException as exc:
                failure = exc
                raise
    except Exception as exc:
        if failure is not None:
            raise failure from exc
        disable(exc)
        if not entered:
            yield lambda step: None


@contextmanager
def _tracked_sample(
    sample_dir, *, directory, args, config, upload_disabled, on_failure
):
    tracker = MLflowTracker()
    sample_index = int(sample_dir.name)
    run_name = sample_run_name(directory, args, sample_index)
    params = {**config, "sample_index": sample_index}
    tags = {
        "experiment_type": LATENT_TRAJECTORY,
        "model": args.model_name,
        "task": args.task,
        "method": "latent_mas",
        "recording": "saved_results",
        "data_dir": str(sample_dir.resolve()),
        "state_dir": str(directory.resolve()),
        "sample_id": sample_dir.name,
        "sample_index": str(sample_index),
        "latent_space_realign": str(args.latent_space_realign),
        "rel_dir": f"{LATENT_TRAJECTORY}/{args.task}/{args.model_name}/{directory.name}/samples/{sample_dir.name}",
    }
    try:
        tracker.start_staging_run(
            experiment_name=mlflow_experiment_name(LATENT_TRAJECTORY),
            run_name=run_name,
            tags=tags,
        )
    except BaseException:
        with suppress(Exception):
            tracker.discard_staging_run(status="FAILED")
        raise

    def save_reference(run_id):
        save_json_atomic(
            sample_dir / "tracking.json",
            {
                "run_id": run_id,
                "experiment_name": mlflow_experiment_name(LATENT_TRAJECTORY),
                "tracking_uri": tracker.tracking_uri,
            },
        )

    futures = {}
    status = "FAILED"
    failure = None
    try:
        tracker.log_params(params)
        save_reference(tracker.active_run_id)
        with ThreadPoolExecutor(max_workers=1) as uploads:

            def upload(step):
                if upload_disabled.is_set():
                    return None
                try:
                    return publish_step(tracker, args, sample_dir, step)
                except Exception as exc:
                    on_failure(exc)
                    return None

            def submit(step):
                if upload_disabled.is_set():
                    return
                for future in futures.values():
                    if future.done():
                        future.result()
                key = step
                ready = all(
                    is_readout_complete(
                        sample_dir / "continuations" / c / f"step_{step:03d}.json"
                    )
                    for c in args.conditions
                )
                if key not in futures and ready:
                    futures[key] = uploads.submit(upload, step)

            try:
                yield submit
                status = "FINISHED"
            except (Exception, KeyboardInterrupt) as exc:
                failure = exc
                save_json_atomic(
                    sample_dir / "failure.json",
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            finally:
                for step in range(args.latent_step_start, args.latent_steps + 1):
                    submit(step)
                totals = {
                    "completed_continuations": 0,
                    "recorded_samples": int(bool(futures)),
                    "completed_steps": len(futures),
                    "correct_continuations": 0,
                }
                history = []
                for future in futures.values():
                    result = future.result()
                    if result is None:
                        continue
                    history.extend(result.pop("history"))
                    for key, value in result.items():
                        totals[key] += value
        if upload_disabled.is_set():
            with suppress(Exception):
                tracker.discard_staging_run(status="FAILED")
            if failure is not None:
                raise failure
            return
        totals["accuracy"] = totals["correct_continuations"] / max(
            totals["completed_continuations"], 1
        )
        records = []
        for step in futures:
            reference = json.loads(
                (sample_dir / "traces" / f"step_{step:03d}.json").read_text()
            )
            for condition in args.conditions:
                row = json.loads(
                    (
                        sample_dir
                        / "continuations"
                        / condition
                        / f"step_{step:03d}.json"
                    ).read_text()
                )
                records.append(
                    {
                        **row,
                        "sample_index": int(sample_dir.name),
                        "sample_key": sample_dir.name,
                        "trace_id": reference["trace_id"],
                        "generated_length": row["content_length"],
                        "generated_text": row["evaluation_text"],
                    }
                )
        summary = build_summary_dict(
            records, args.conditions, totals["recorded_samples"], params
        )
        summary["status"] = "complete" if status == "FINISHED" else "failed"
        summary["metrics"].update(totals)
        save_experiment_summary(sample_dir, summary, records)
        totals = summary["metrics"]
        save_json_atomic(
            sample_dir / "invocation.json",
            {
                "conditions": args.conditions,
                "sample_index": sample_index,
                "requested_sample_indices": getattr(args, "sample_indices", None),
            },
        )
        artifacts = [
            (directory / name, None)
            for name in ("metadata.json", "state.json")
            if (directory / name).is_file()
        ]
        for path in sorted(sample_dir.rglob("*")):
            if path.is_file() and ".tmp." not in path.name:
                relative = path.relative_to(sample_dir)
                artifacts.append(
                    (path, str(relative.parent) if len(relative.parts) > 1 else None)
                )
        run = tracker.promote_staging_run(
            final_run_name=run_name,
            tags=tags,
            metrics=totals,
            params=params,
            artifacts=artifacts,
            status=status,
        )
        save_reference(run.info.run_id)
        for offset in range(0, len(history), 1000):
            tracker.client.log_batch(
                run.info.run_id, metrics=history[offset : offset + 1000]
            )
        tracker.client.log_artifact(run.info.run_id, str(sample_dir / "tracking.json"))
    except BaseException:
        with suppress(Exception):
            tracker.discard_staging_run(status="FAILED")
        raise
    if failure is not None:
        raise failure


def sample_run_name(directory, args, sample_index):
    alignment = "on" if args.latent_space_realign else "off"
    return f"{args.task}_{args.model_name.split('/')[-1]}_align-{alignment}_sample-{sample_index}_{directory.name[:8]}"
