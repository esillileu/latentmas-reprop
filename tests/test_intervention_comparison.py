"""Four-condition calculations, prompt parity, cache continuity and resumption."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_intervention_continuation import _build_mock_wrapper

from latentmas_reprop.application.intervention_continuation.conditions import (
    COMPARISON_CONDITIONS,
)
from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
    generation_settings,
)
from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.domain.services.prompts.single_agent import (
    build_agent_messages_single_agent,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage, ResearchDataReader
from src.run.cli import parse_run_matrix


def comparison_case(tmp_path, tracker=None):
    wrapper = _build_mock_wrapper()

    def prepare(messages, **kwargs):
        text = "\n".join(message["content"] for message in messages[0])
        ids = [ord(char) % 32 for char in text]
        return (
            [text],
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            [[str(i) for i in ids]],
        )

    wrapper.prepare_chat_batch = prepare
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    dataset = SimpleNamespace(
        load=lambda **kwargs: [
            {"question": "Q1", "gold": "1"},
            {"question": "Q2", "gold": "2"},
        ]
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    case = InterventionContinuationUseCase(
        dataset_port=dataset,
        evaluator_port=evaluator,
        storage=FileStorage(PathResolver(tmp_path)),
        tracker_port=tracker,
    )
    args = SimpleNamespace(
        model_name="Qwen/mock",
        task="gsm8k",
        split="test",
        method="latent_mas",
        latent_steps=2,
        intervention_latent_steps=1,
        max_new_tokens=1,
        temperature=0.0,
        top_p=0.95,
        top_k=20,
        seed=42,
        max_samples=1,
        no_progress=True,
        intervention_conditions=list(COMPARISON_CONDITIONS),
    )
    return case, wrapper, args


def test_four_conditions_preserve_prompts_and_accumulate_full_kv(tmp_path):
    case, wrapper, args = comparison_case(tmp_path)
    summary, records = case.execute(wrapper, args)
    assert summary["conditions"] == list(COMPARISON_CONDITIONS)
    assert (
        wrapper.model.forward.call_count == 13
    )  # two prefills, 2+3 latent, six role prefills
    by_condition = {record["condition"]: record for record in records}
    assert [record["latent_steps_executed"] for record in records] == [0, 2, 0, 3]
    assert {record["sample_id"] for record in records} == {"000000"}
    assert len({record["rng_provenance"]["seed"] for record in records}) == 1
    baseline_prompt = wrapper.prepare_chat_batch(
        [build_agent_messages_single_agent("Q1", args)]
    )[0][0]
    assert by_condition["baseline"]["initial_prompt"] == baseline_prompt
    assert by_condition["latent_only"]["initial_prompt"] == baseline_prompt
    for condition in ["intervention_only", "latent_intervention"]:
        record = by_condition[condition]
        steps = record["identity"]["step"]
        prompt_len = len(record["initial_prompt_token_ids"])
        length = prompt_len + steps
        assert [stage["role"] for stage in record["stages"]] == [
            "critic",
            "refiner",
            "judger",
        ]
        for stage in record["stages"]:
            assert stage["cache_length_before"] == length
            assert "Q1" in stage["prompt"]
            length += len(stage["token_ids"]) + stage["latent_steps"]
            assert stage["cache_length_after"] == length
        assert (
            sum(stage["latent_steps"] for stage in record["stages"]) + steps
            == record["latent_steps_executed"]
        )
    assert (
        len(
            list(
                case.storage.base_dir.glob("intervention_continuation/*/*/*/state.json")
            )
        )
        == 3
    )
    wrapper.model.forward.reset_mock()
    assert case.execute(wrapper, args)[1] == records
    assert wrapper.model.forward.call_count == 0


def test_comparison_extends_conditions_and_reads_each_state(tmp_path):
    case, wrapper, args = comparison_case(tmp_path)
    args.intervention_conditions = ["baseline"]
    case.execute(wrapper, args)
    args.intervention_conditions = ["intervention_only"]
    summary, _ = case.execute(wrapper, args)
    assert summary["conditions"] == ["baseline", "intervention_only"]
    config = summary["config"]
    assert len(config["result_references"]["000000"]) == 2
    comparison_root = case.storage.base_dir / config["state_path"]
    run_dir = next(path.parent for path in comparison_root.glob("runs/*/config.json"))
    run = str(run_dir.relative_to(case.storage.base_dir))
    reader = ResearchDataReader(
        tensor_store=case.storage, path_resolver=case.storage._resolver
    )
    with pytest.raises(ValueError, match="condition is required"):
        reader.load_intervention_state(run, "000000")
    for condition in ["baseline", "intervention_only"]:
        _, _, meta = reader.load_intervention_state(run, "000000", condition=condition)
        assert meta["question"] == "Q1"
        reference = next(
            path
            for path in config["result_references"]["000000"]
            if path.endswith("/" + condition)
        )
        result_hash = reference.split("/")[-2]
        hidden, record = reader.load_intervention_result(
            run, "000000", 0, result_hash, condition
        )
        assert hidden.shape[0] > 0 and record["condition"] == condition
    args.intervention_conditions = ["baseline"]
    args.top_k = 1
    wrapper.model.forward.reset_mock()
    summary, records = case.execute(wrapper, args)
    assert records[0]["identity"]["top_k"] == 1
    assert summary["conditions"] == ["baseline"]


def test_condition_order_and_sample_extension_do_not_change_results(tmp_path):
    case, wrapper, args = comparison_case(tmp_path / "first")
    _, first = case.execute(wrapper, args)
    args.max_samples = 2
    args.intervention_conditions = list(reversed(COMPARISON_CONDITIONS))
    _, extended = case.execute(wrapper, args)
    assert {r["condition"]: r for r in first} == {
        r["condition"]: r for r in extended if r["sample_index"] == 0
    }
    fresh_case, _, _ = comparison_case(tmp_path / "fresh")
    # Use the same frozen model to compare RNG and cache handling.
    _, fresh = fresh_case.execute(wrapper, args)
    assert extended == fresh


@pytest.mark.parametrize("filter_args", [{"top_k": 1}, {"top_p": 0.5}])
def test_sampling_filters_and_multiple_eos(filter_args, monkeypatch):
    wrapper = _build_mock_wrapper()
    monkeypatch.setattr(
        wrapper.model.lm_head,
        "forward",
        lambda hidden: torch.tensor([3.0, 1.0, 0.0, -1.0]),
    )
    seen = []

    def sample(probs, num_samples):
        seen.append(probs)
        return torch.tensor([int(probs.argmax())])

    monkeypatch.setattr(torch, "multinomial", sample)
    ids, hidden, reason, exhausted = decode_tokens_with_hiddens(
        wrapper.model,
        wrapper.device,
        None,
        torch.ones(8),
        wrapper.tokenizer,
        1,
        temperature=1,
        report_progress=False,
        **filter_args,
    )
    assert ids == [0] and len(hidden) == 1 and exhausted and reason == "length"
    assert torch.equal(seen[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    wrapper.model.generation_config = SimpleNamespace(eos_token_id=[0, 1])
    ids, hidden, reason, exhausted = decode_tokens_with_hiddens(
        wrapper.model,
        wrapper.device,
        None,
        torch.ones(8),
        wrapper.tokenizer,
        1,
        report_progress=False,
    )
    assert ids == hidden == [] and reason == "stop" and not exhausted


def test_new_preset_is_one_run_and_does_not_change_existing_preset():
    runs = parse_run_matrix(["-c", "intervention_continuation/gsm8k_q314", "--dry-run"])
    assert len(runs) == 1
    args = runs[0]
    assert args.intervention_conditions == list(COMPARISON_CONDITIONS)
    assert args.latent_steps == 60 and args.intervention_latent_steps == 20
    assert args.model_revision == "40c069824f4251a91eefaf281ebe4c544efd3e18"
    existing = parse_run_matrix(["-c", "intervention_continuation/gsm8k_q38"])[0]
    assert (
        existing.initial_role == "direct_solve"
        and existing.intervention_conditions is None
    )
    wrapper = _build_mock_wrapper()
    wrapper.model.generation_config = SimpleNamespace(eos_token_id=[99, 98])
    assert generation_settings(wrapper, existing) == {
        "temperature": 0.6,
        "top_p": 1.0,
        "top_k": 0,
        "eos_ids": [99],
    }
    assert generation_settings(wrapper, args) == {
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "eos_ids": [99, 98],
    }


@pytest.mark.parametrize(
    "override",
    [
        ["--intervention_conditions", "baseline,rqe"],
        ["--prompt", "hierarchical"],
        ["--think"],
        ["--use_vllm"],
        ["--intervention_latent_steps", "-1"],
        ["--max_new_tokens", "0"],
        ["--top_p", "0"],
        ["--top_k", "-1"],
        ["--initial_input", "q"],
    ],
)
def test_invalid_comparison_settings_fail_before_loading_model(override):
    with pytest.raises(SystemExit):
        parse_run_matrix(["-c", "intervention_continuation/gsm8k_q314", *override])
