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
    from latentmas_reprop.domain.services.prompts import build_initial_messages

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
    args.intervention_conditions = [
        "rqe",
        "rqle",
        "rqlce",
        "rqlre",
        "rqlqe",
        "rqlrqe",
        "rqrqe",
    ]
    _, results = uc.execute(wrapper, args)
    records = {record["condition_name"]: record for record in results}
    assert set(records) == {"RQE", "RQLE", "RQLCE", "RQLRE", "RQLQE", "RQLRQE", "RQRQE"}
    initial_prompt = prepare([build_initial_messages(item["question"], args)])[0][0]
    for record in records.values():
        assert record["initial_role"] == "planner"
        assert record["initial_prompt"] == initial_prompt
        assert record["initial_prompt_token_ids"] == [ord(c) for c in initial_prompt]
    for name, condition in [
        ("RQLRE", "rqlre"),
        ("RQLQE", "rqlqe"),
        ("RQLRQE", "rqlrqe"),
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
    args.intervention_conditions = ["rqlrqe", "RQLRQE"]
    _, aliases = uc.execute(wrapper, args)
    assert len(aliases) == 1
    assert aliases[0]["identity"] == records["RQLRQE"]["identity"]
    assert aliases[0]["generated_token_ids"] == records["RQLRQE"]["generated_token_ids"]
    assert wrapper.model.forward.call_count == 0
    args.intervention_conditions = None
    args.initial_input = "q"
    _, q_results = uc.execute(wrapper, args)
    q_records = {record["condition_name"]: record for record in q_results}
    assert set(q_records) == {"QE", "QLE"}
    assert q_records["QE"]["initial_prompt"] != initial_prompt
    assert q_records["QE"]["identity"]["step"] == 0
    assert q_records["QLE"]["identity"]["step"] == 2


def test_rng_is_sample_based_and_independent_of_order_and_reuse(tmp_path, monkeypatch):
    import torch

    from latentmas_reprop.application.intervention_continuation.runner import (
        InterventionContinuationRunner,
    )

    original = InterventionContinuationRunner.execute_unprompted

    def decode(self, *args):
        hidden, record = original(self, *args)
        record["draw"] = torch.rand(1).item()
        return hidden, record

    monkeypatch.setattr(InterventionContinuationRunner, "execute_unprompted", decode)
    wrapper = _build_mock_wrapper()
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
        intervention_conditions=["rqe", "rqle"],
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)

    def run(root):
        uc = InterventionContinuationUseCase(
            dataset_port=dataset,
            evaluator_port=evaluator,
            storage=FileStorage(PathResolver(root)),
        )
        return uc.execute(wrapper, args)[1]

    first = run(tmp_path / "first")
    assert first[0]["rng_provenance"]["seed"] == first[1]["rng_provenance"]["seed"]
    assert first[0]["draw"] == first[1]["draw"]
    args.intervention_conditions = ["rqle"]
    run(tmp_path / "second")
    args.intervention_conditions = ["rqle", "rqe"]
    second = run(tmp_path / "second")
    assert {r["condition"]: r["draw"] for r in first} == {
        r["condition"]: r["draw"] for r in second
    }
    args.latent_steps = 3
    args.max_new_tokens = 2
    third = run(tmp_path / "second")
    assert third[0]["rng_provenance"]["seed"] == first[0]["rng_provenance"]["seed"]
    assert all(r["identity"]["rng_policy"] == "sample_v2" for r in third)


def test_configured_initial_roles_execute_conditions(tmp_path):
    from latentmas_reprop.domain.services.prompts import load_intervention_config

    wrapper = _build_mock_wrapper()
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    uc = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=FileStorage(PathResolver(tmp_path)),
    )
    for variant in [*load_intervention_config()["roles"], None]:
        conditions = (
            ["qe", "qle"] if variant is None else ["rqe", "rqle", "rqrqe", "rqlrqe"]
        )
        args = SimpleNamespace(
            task="gsm8k",
            model_name="Qwen/mock",
            latent_steps=1,
            max_new_tokens=1,
            temperature=0.0,
            top_p=1.0,
            seed=42,
            max_samples=1,
            no_progress=True,
            initial_input="q" if variant is None else "rq",
            initial_role=variant or "planner",
            intervention_conditions=conditions,
        )
        _, records = uc.execute(wrapper, args)
        assert [record["condition"] for record in records] == conditions
        assert all(
            record["identity"]["condition"] == record["condition"] for record in records
        )
        assert all(
            record["identity"]["rng_policy"] == "sample_v2" for record in records
        )
        assert all(record["initial_prompt"] for record in records)
