"""Verify trajectory MLflow runs, saved-result traces and artifact uploads."""

from threading import Event, get_ident
from types import SimpleNamespace

import mlflow
import pytest
import torch
from mlflow.tracking import MlflowClient

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import save_json_atomic, save_tensor
from latentmas_reprop.infrastructure.tracking import MLflowTracker
from src.run import latent_trajectory_tracking as tracking


@pytest.mark.parametrize("fail", [False, True])
def test_tracking_exports_results_and_failure(
    tmp_path, monkeypatch, make_mlflow_db, fail
):
    previous_uri = mlflow.get_tracking_uri()
    tracker = MLflowTracker(
        tracking_uri=make_mlflow_db(tmp_path / "tracking.db"),
        artifact_location=str(tmp_path / "artifacts"),
        path_resolver=PathResolver(tmp_path),
    )
    monkeypatch.setattr(tracking, "MLflowTracker", lambda: tracker)
    previews = []
    update_trace = tracker.update_current_trace

    def capture_preview(**kwargs):
        previews.append(kwargs.get("response_preview", ""))
        update_trace(**kwargs)

    monkeypatch.setattr(tracker, "update_current_trace", capture_preview)
    directory = tmp_path / "results"
    sample = directory / "samples/000000"
    save_json_atomic(
        directory / "samples/000037/sample.json", {"question": "unselected"}
    )
    states = {
        "without_think": {
            "prompt": "prompt",
            "prompt_token_ids": [1, 2, 3],
            "prompt_length": 3,
            "latent_steps": 3,
        },
        "with_think": {
            "prompt": "prompt<think>",
            "prompt_token_ids": [1, 2, 3, 20],
            "prompt_length": 4,
            "latent_steps": 3,
        },
    }
    save_json_atomic(
        sample / "sample.json",
        {
            "question": "2+2?",
            "gold": "4",
            "index": 0,
            "initial_states": states,
        },
    )
    row = {
        "condition": "a",
        "step": 1,
        "correct": True,
        "generated_length": 3,
        "generated_text": "4",
        "prediction": "4",
        "inserted_prefix_ids": [],
        "inserted_readout_ids": [],
        "status": "complete",
        "files": ["hidden.pt"],
        "evaluation_text": "4",
        "content_length": 3,
        "rng_provenance": {"policy": "sample_v2", "seed": 42},
        "initial_role": "direct_solve",
        "initial_prompt": "prompt",
        "initial_prompt_token_ids": [1, 2, 3],
        "latent_steps_executed": 1,
        "intervention_length": 0,
        "identity": {
            "sampling": {"temperature": 0.6},
            "max_new_tokens": 2048,
            "decoder": "test-decoder",
        },
    }
    for step in (1, 2):
        for condition in "abcd":
            save_json_atomic(
                sample / f"continuations/{condition}/step_{step:03d}.json",
                {
                    **row,
                    "step": step,
                    "condition": condition,
                    "correct": condition in "ac",
                    "initial_prompt_token_ids": [1, 2, 3]
                    + ([20] if condition in "bd" else []),
                    "intervention_length": int(condition in "cd"),
                    "latent_steps_executed": step,
                },
            )
            save_tensor(
                sample / f"continuations/{condition}/step_{step:03d}/hidden.pt",
                torch.zeros(3, 8),
            )
    # An interrupted third step must not publish a partial four-condition trace.
    save_json_atomic(sample / "continuations/a/step_003.json", {**row, "step": 3})
    args = SimpleNamespace(
        model_name="Qwen/Qwen3-14B",
        task="gsm8k",
        latent_space_realign=False,
        conditions=list("abcd"),
        max_samples=1,
        latent_step_start=1,
        latent_steps=3,
    )

    def execute():
        started = Event()
        main_thread = get_ident()
        publish = tracking.publish_step

        def background_publish(*a, **kw):
            assert get_ident() != main_thread
            result = publish(*a, **kw)
            started.set()
            return result

        monkeypatch.setattr(tracking, "publish_step", background_publish)
        with tracking.tracked_sample(
            sample, directory=directory, args=args, config={"seed": 42}
        ) as submit:
            submit(1)
            assert started.wait(timeout=20)
            live = tracker.client.search_traces(
                locations=[
                    tracker.client.get_run(tracker.active_run_id).info.experiment_id
                ],
                run_id=tracker.active_run_id,
            )
            assert len(live) == 1
            assert live[0].info.tags["latent_step"] == "1"
            assert len([s for s in live[0].data.spans if s.parent_id is not None]) == 5
            submit(2)
            if fail:
                raise RuntimeError("inference interrupted")

    try:
        if fail:
            with pytest.raises(RuntimeError, match="inference interrupted"):
                execute()
        else:
            execute()
        import json

        run_id = json.loads((sample / "tracking.json").read_text())["run_id"]
        client = MlflowClient(tracking_uri=tracker.tracking_uri)
        run = client.get_run(run_id)
        assert run.info.status == ("FAILED" if fail else "FINISHED")
        assert run.data.metrics["completed_continuations"] == 8
        assert run.data.metrics["completed_steps"] == 2
        assert run.data.metrics["recorded_samples"] == 1
        assert run.data.metrics["accuracy"] == 0.5
        assert run.data.tags["promoted_from"]
        assert (
            client.get_metric_history(run_id, "samples/000000/a/generated_length")[
                0
            ].step
            == 1
        )
        traces = client.search_traces(locations=[run.info.experiment_id], run_id=run_id)
        assert len(traces) == 2
        assert {t.info.tags["latent_step"] for t in traces} == {"1", "2"}
        first = next(t for t in traces if t.info.tags["latent_step"] == "1")
        assessments = {a.name: a for a in first.info.assessments}
        assert assessments["expected_answer"].expectation.value == "4"
        for condition in "abcd":
            assert assessments[f"{condition}_correct"].feedback.value == (
                condition in "ac"
            )
        root = next(s for s in first.data.spans if s.parent_id is None)
        assert root.outputs["a_correct"] is True
        assert root.outputs["b_correct"] is False
        assert root.outputs["a_prediction"] == "4"
        assert any(
            "a: 4 (PASS)" in preview and "b: 4 (FAIL)" in preview
            for preview in previews
        )
        child = next(s for s in first.data.spans if s.name == "decode_a")
        assert child.outputs["generated_text"] == "4"
        assert child.inputs["initial_prompt_token_ids"] == [1, 2, 3]
        assert child.inputs["rng_provenance"]["policy"] == "sample_v2"
        assert child.inputs["sampling"]["temperature"] == 0.6
        assert "mlflow.chat.tokenUsage" in child.attributes
        initial = next(s for s in first.data.spans if s.name == "initial_states")
        assert initial.inputs["initial_states"]["with_think"]["prompt"].endswith(
            "<think>"
        )
        assert initial.outputs["prefills"]["with_think"]["latent_steps"] == 1
        assert first.info.tags["result_fingerprint"]
        assert first.info.tags["model"] == args.model_name
        files = client.list_artifacts(run_id, "continuations/a")
        assert any(f.path.endswith("step_001.json") for f in files)
        assert client.list_artifacts(run_id, "samples") == []
        assert run.data.tags["sample_id"] == "000000"
        assert "_sample-0_" in run.info.run_name
        assert run.data.metrics["a_accuracy"] == 1
        assert run.data.metrics["b_accuracy"] == 0
        assert any(a.path == "summary.json" for a in client.list_artifacts(run_id))
        if not fail:
            original = {t.info.tags["latent_step"]: t.info.trace_id for t in traces}
            with tracking.tracked_sample(
                sample, directory=directory, args=args, config={"seed": 42}
            ) as submit:
                submit(1)
                submit(2)
            retained = client.search_traces(
                locations=[run.info.experiment_id], run_id=run_id
            )
            assert {
                t.info.tags["latent_step"]: t.info.trace_id for t in retained
            } == original
            changed = sample / "continuations/b/step_001.json"
            record = json.loads(changed.read_text())
            save_json_atomic(changed, {**record, "correct": True})
            with tracking.tracked_sample(
                sample, directory=directory, args=args, config={"seed": 42}
            ) as submit:
                submit(1)
            replaced = client.search_traces(
                locations=[run.info.experiment_id], run_id=run_id
            )
            latest = {t.info.tags["latent_step"]: t.info.trace_id for t in replaced}
            assert len(replaced) == 2
            assert latest["1"] != original["1"]
            assert latest["2"] == original["2"]
            from shutil import copytree

            second = directory / "samples/000037"
            copytree(sample, second, dirs_exist_ok=True)
            metadata = json.loads((second / "sample.json").read_text())
            save_json_atomic(second / "sample.json", {**metadata, "index": 37})
            with tracking.tracked_sample(
                second, directory=directory, args=args, config={"seed": 42}
            ) as submit:
                submit(1)
            second_id = json.loads((second / "tracking.json").read_text())["run_id"]
            assert second_id != run_id
            second_run = client.get_run(second_id)
            assert second_run.data.params["sample_index"] == "37"
            assert "_sample-37_" in second_run.info.run_name
            second_traces = client.search_traces(
                locations=[run.info.experiment_id], run_id=second_id
            )
            assert len(second_traces) == 2
            assert {t.info.tags["dataset_sample_key"] for t in second_traces} == {
                "000037"
            }
            first_traces = client.search_traces(
                locations=[run.info.experiment_id], run_id=run_id
            )
            assert {t.info.tags["dataset_sample_key"] for t in first_traces} == {
                "000000"
            }
            # Append step zero to the same canonical run and retain existing traces.
            from shutil import copyfile

            args.latent_step_start = 0
            for condition in "abcd":
                source = sample / f"continuations/{condition}/step_001.json"
                target = source.with_name("step_000.json")
                row = json.loads(source.read_text())
                save_json_atomic(target, {**row, "step": 0, "latent_steps_executed": 0})
                target.with_suffix("").mkdir()
                copyfile(
                    source.with_suffix("") / "hidden.pt",
                    target.with_suffix("") / "hidden.pt",
                )
            with tracking.tracked_sample(
                sample, directory=directory, args=args, config={"seed": 42}
            ) as submit:
                submit(0)
            assert (
                json.loads((sample / "tracking.json").read_text())["run_id"] == run_id
            )
            extended = client.search_traces(
                locations=[run.info.experiment_id], run_id=run_id
            )
            identities = {t.info.tags["latent_step"]: t.info.trace_id for t in extended}
            assert set(identities) == {"0", "1", "2"}
            assert {k: identities[k] for k in latest} == latest
            assert client.get_run(run_id).data.metrics["completed_continuations"] == 12
            assert client.get_run(run_id).data.metrics["completed_steps"] == 3
        if fail:
            assert any(a.path == "failure.json" for a in client.list_artifacts(run_id))
    finally:
        mlflow.set_tracking_uri(previous_uri)
