"""Publication failures must not interrupt local collection."""

from threading import Event
from types import SimpleNamespace

import pytest

from src.run import latent_trajectory_tracking as tracking


@pytest.mark.parametrize("stage", ["setup", "upload", "promotion"])
def test_upload_failure_keeps_computing_and_disables_later_samples(
    tmp_path, monkeypatch, stage
):
    import json
    from unittest.mock import Mock

    disabled = Event()
    tracker = Mock()
    tracker.active_run_id = "staging-run"
    tracker.tracking_uri = "test"
    error = RuntimeError("max retries exceeded")
    if stage == "setup":
        tracker.start_staging_run.side_effect = error
    elif stage == "promotion":
        tracker.promote_staging_run.side_effect = error
    factory = Mock(return_value=tracker)
    monkeypatch.setattr(tracking, "MLflowTracker", factory)
    monkeypatch.setattr(tracking, "is_readout_complete", lambda path: stage == "upload")
    publish = Mock(side_effect=error)
    monkeypatch.setattr(tracking, "publish_step", publish)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="test/model",
        latent_space_realign=False,
        conditions=["a"],
        latent_step_start=0,
        latent_steps=2,
    )
    progress = []
    sample = tmp_path / "samples/000000"
    with tracking.tracked_sample(
        sample, directory=tmp_path, args=args, config={}, upload_disabled=disabled
    ) as submit:
        for step in range(3):
            progress.append(step)
            submit(step)
            if stage == "upload" and step == 0:
                assert disabled.wait(timeout=5)
    assert progress == [0, 1, 2]
    assert disabled.is_set()
    assert json.loads((sample / "upload_failure.json").read_text())["message"] == str(
        error
    )
    if stage == "upload":
        assert publish.call_count == 1
    tracker.discard_staging_run.side_effect = RuntimeError("cleanup unavailable")
    with tracking.tracked_sample(
        tmp_path / "samples/000001",
        directory=tmp_path,
        args=args,
        config={},
        upload_disabled=disabled,
    ) as submit:
        submit(0)
        progress.append(3)
    assert progress == [0, 1, 2, 3]
    assert factory.call_count == 1


def test_disabled_uploads_preserve_inference_errors(tmp_path):
    disabled = Event()
    disabled.set()
    with (
        pytest.raises(RuntimeError, match="inference failed"),
        tracking.tracked_sample(
            tmp_path, directory=tmp_path, args=None, config={}, upload_disabled=disabled
        ),
    ):
        raise RuntimeError("inference failed")
