from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from latentmas_reprop.application.receiver_reasoning_use_case import (
    ReceiverReasoningUseCase,
)
from src.run.cli import parse_args


def test_paired_matrix_reuses_pristine_context(tmp_path):
    args = parse_args(["-c", "lmas/receiver_reasoning/gsm8k"])
    args.max_samples = 2
    built = []
    decoded = []

    class Method:
        def __init__(self, model, **kwargs):
            self.step = kwargs["latent_steps"]

        def build_latent_contexts(self, items):
            context = ((torch.full((1, 1, self.step, 1), float(self.step)),),)
            built.append((items[0]["question"], self.step, context))
            return context, [[]]

        def decode_with_context(self, items, **kwargs):
            context = kwargs["past_kv"]
            assert torch.all(context[0][0] == self.step)
            decoded.append(
                (items[0]["question"], self.step, kwargs["receiver_mode"], context)
            )
            context[0][0].zero_()
            correct = kwargs["receiver_mode"] == "free" or self.step == 40
            return [
                {
                    "prediction": "1" if correct else "0",
                    "raw_prediction": "1",
                    "correct": correct,
                    "agents": [{"generated_tokens": 3}],
                }
            ]

    dataset = SimpleNamespace(
        load=lambda **kwargs: [
            {"question": "a", "gold": "1"},
            {"question": "b", "gold": "1"},
        ]
    )
    cache = SimpleNamespace(get_layer_path=lambda layer: tmp_path)
    with patch(
        "latentmas_reprop.application.receiver_reasoning_use_case.LatentMASMethod",
        Method,
    ):
        summary, records = ReceiverReasoningUseCase(
            dataset_port=dataset, cache_port=cache
        ).execute(None, args)
    assert len(built) == 4
    assert len(decoded) == len(records) == 8
    assert all(torch.all(context[0][0] == step) for _, step, context in built)
    assert summary["substitution_signal"] == 1
    assert (
        summary["upstream_levels"]["10"]["paired_correctness"]["free_only_correct"] == 2
    )
    assert summary["upstream_levels"]["40"]["cells"]["answer_only"][
        "prediction_frequencies"
    ] == {"1": 2}
    assert len(list(tmp_path.glob("*/sample_results.jsonl"))) == 1


@pytest.mark.parametrize("steps", ["40,10", "10,10", "0,40", "10", "a,40"])
def test_invalid_matrix(steps):
    with pytest.raises(SystemExit):
        parse_args(["-c", "lmas/receiver_reasoning/gsm8k", "--upstream_steps", steps])


def test_answer_only_disables_thinking_and_prefills_answer():
    from latentmas_reprop.domain.services.latent_mas import LatentMASMethod

    class Tokenizer:
        def __call__(self, prompts, **kwargs):
            assert all(p.endswith(r"\boxed{") for p in prompts)
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
            }

        def convert_ids_to_tokens(self, ids):
            return ["token"] * len(ids)

    class Model:
        device = "cpu"
        tokenizer = Tokenizer()

        def prepare_chat_batch(self, messages, **kwargs):
            assert kwargs["chat_template_kwargs"] == {"enable_thinking": False}
            assert "reason step" not in messages[0][-1]["content"]
            return ["assistant\n<think>\n\n</think>\n\n"], None, None, None

        def generate_text_batch(self, ids, mask, **kwargs):
            assert kwargs["max_new_tokens"] == 64
            return (
                [
                    "18}",
                ],
                None,
                [2],
            )

    args = SimpleNamespace(task="gsm8k", model_name="Qwen/Qwen3-0.6B", think=True)
    method = LatentMASMethod(Model(), args=args)
    result = method.decode_with_context(
        [{"question": "question", "gold": "18"}],
        receiver_mode="answer_only",
        max_new_tokens=64,
    )[0]
    assert result["raw_prediction"] == r"\boxed{18}"
    assert result["correct"]
    assert result["agents"][-1]["generated_tokens"] == 2
    assert not result["agents"][-1]["input"].endswith("<think>")
