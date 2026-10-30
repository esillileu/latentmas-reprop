"""Condition order, prompt content and checkpoint regression coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage


def test_rqrqe_branches_from_prefill_and_reuses_result(tmp_path):
    wrapper = _build_mock_wrapper()
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=7,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_samples=1,
        no_progress=True,
        intervention_conditions=["RQRQE"],
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    uc = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=FileStorage(PathResolver(tmp_path)),
    )
    _, records = uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 2  # prefill and P2, no latent forwards
    assert records[0]["identity"]["step"] == 0
    assert records[0]["condition_name"] == "RQRQE"
    wrapper.model.forward.reset_mock()
    uc.execute(wrapper, args)
    assert wrapper.model.forward.call_count == 0


def test_nine_conditions_use_actual_prompt_tokens(tmp_path):
    import torch

    from latentmas_reprop.application.intervention_continuation.identity import (
        intervention_messages,
    )
    from latentmas_reprop.domain.services.prompts import build_intervention_p1_messages

    wrapper = _build_mock_wrapper()

    def prepare(batch, **kwargs):
        text = "\n".join(message["content"] for message in batch[0])
        ids = [ord(c) for c in text]
        return (
            [text],
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            [[str(i) for i in ids]],
        )

    wrapper.prepare_chat_batch = prepare
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/mock",
        latent_steps=2,
        max_new_tokens=1,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_samples=1,
        no_progress=True,
    )
    item = {"question": "A distinctive problem?", "gold": "1"}
    dataset = MagicMock()
    dataset.load.return_value = [item]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    uc = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=FileStorage(PathResolver(tmp_path)),
    )
    _, results = uc.execute(wrapper, args)
    records = {record["condition_name"]: record for record in results}
    assert set(records) == {"RQE", "RQLE", "RQLCE", "RQLRE", "RQLQE", "RQLRQE", "RQRQE"}
    p1 = prepare([build_intervention_p1_messages(item["question"], args)])[0][0]
    for record in records.values():
        assert record["initial_role"] == "planner"
        assert record["initial_prompt"] == p1
        assert record["initial_prompt_token_ids"] == [ord(c) for c in p1]
    for name, condition in [
        ("RQLRE", "rqlre"),
        ("RQLQE", "rqlqe"),
        ("RQLRQE", "plpe"),
        ("RQRQE", "rqrqe"),
    ]:
        prompt = prepare([intervention_messages(condition, item, args)])[0][0]
        assert records[name]["intervention_prompt"] == prompt
        assert records[name]["intervention_token_ids"] == [ord(c) for c in prompt]
    assert item["question"] not in records["RQLRE"]["intervention_prompt"]
    assert "Judger" not in records["RQLQE"]["intervention_prompt"]
    assert (
        records["RQLRQE"]["intervention_prompt"]
        == records["RQRQE"]["intervention_prompt"]
    )
    assert records["RQLCE"]["intervention_token_ids"] == [10, 11]
    assert (
        records["RQE"]["intervention_token_ids"]
        == records["RQLE"]["intervention_token_ids"]
        == []
    )
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    args.intervention_conditions = ["plpe", "RQLRQE"]
    _, aliases = uc.execute(wrapper, args)
    assert len(aliases) == 1
    assert aliases[0]["identity"] == records["RQLRQE"]["identity"]
    assert aliases[0]["generated_token_ids"] == records["RQLRQE"]["generated_token_ids"]
    assert wrapper.model.forward.call_count == 0
    args.intervention_conditions = None
    args.p1_variant = "question_only"
    _, q_results = uc.execute(wrapper, args)
    q_records = {record["condition_name"]: record for record in q_results}
    assert set(q_records) == {"QE", "QLE"}
    assert q_records["QE"]["initial_prompt"] != p1
    assert q_records["QE"]["identity"]["step"] == 0
    assert q_records["QLE"]["identity"]["step"] == 2
