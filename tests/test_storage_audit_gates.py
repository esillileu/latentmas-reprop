"""Audit gate tests for storage refactoring verification."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from latentmas_reprop.application.receiver_acquisition.runner import (
    run_acquisition_loop,
)
from latentmas_reprop.application.receiver_acquisition.sampling import SecretDigitSample
from latentmas_reprop.infrastructure.paths.resolver import (
    PathResolver,
    get_path_resolver,
)
from latentmas_reprop.infrastructure.storage import (
    ResearchDataReader,
    save_condition_result,
)


def test_acquisition_lazy_cache_build():
    """Acquisition loop builds sender cache only when needed."""
    resolver = get_path_resolver()
    samples = [
        SecretDigitSample(sample_id="s0", sample_index=0, sample_key="s0", digit=1),
        SecretDigitSample(sample_id="s1", sample_index=1, sample_key="s1", digit=2),
    ]
    # Mark s0 complete on disk
    c_dir = resolver.get_condition_dir("run_acq", 0, "full_own")
    save_condition_result(
        c_dir,
        {
            "sample_id": "s0",
            "condition": "own",
            "context_mode": "full",
            "target_digit": 1,
        },
    )

    args = SimpleNamespace(
        acquisition_conditions=["own"],
        context_modes=["full"],
        latent_steps=5,
        save_hidden_states=False,
    )
    model = MagicMock()
    model.device = torch.device("cpu")

    rec_mock = MagicMock()
    rec_mock.hidden_state_artifact_key = None
    rec_mock.to_dict.return_value = {
        "sample_id": "s1",
        "condition": "own",
        "context_mode": "full",
    }
    rec_mock.sample_index = 1
    rec_mock.predicted_digit = 2
    rec_mock.source_probability = 0.9
    rec_mock.content_follow_correct = True
    rec_mock.error = None

    with (
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.build_sender_cache"
        ) as mock_build,
        patch(
            "latentmas_reprop.application.receiver_acquisition.runner.forward_receiver_observation",
            return_value=(rec_mock, None),
        ),
    ):
        bundle_mock = MagicMock()
        bundle_mock.build_latency_sec = 0.05
        bundle_mock.full = None
        mock_build.return_value = bundle_mock

        records, _, _ = run_acquisition_loop(
            model=model,
            args=args,
            samples=samples,
            cross_indices=[0, 0],
            receiver_ids=torch.zeros((1, 2), dtype=torch.long),
            receiver_mask=torch.ones((1, 2), dtype=torch.long),
            mapping={},
            tracker_port=None,
            run_id="run_acq",
        )
        # s0 was complete, so build_sender_cache should only be called once for s1!
        assert mock_build.call_count == 1
        assert len(records) == 2


def test_reader_path_traversal_blocked(tmp_path):
    """Path traversal characters are strictly rejected with ValueError."""
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(path_resolver=resolver)

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_artifact("run_1", "../secret.json")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="../bad", condition="own")

    with pytest.raises(ValueError, match="Path traversal"):
        reader.load_tensor("run_1", "hidden", sample_key="s1", condition="../bad")
