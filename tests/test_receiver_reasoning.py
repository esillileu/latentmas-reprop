from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from latentmas_reprop.application.receiver_reasoning_use_case import (
    ReceiverReasoningUseCase,
)
from src.run.cli import parse_args


@pytest.mark.parametrize(
    "tracked, steps, width",
    [(False, [10, 40], 10), (True, [10, 40], 10), (False, [5], None)],
)
def test_paired_matrix_reuses_pristine_context(tmp_path, tracked, steps, width):
    args = parse_args(["-c", "lmas/receiver_reasoning/gsm8k"])
    args.max_samples = 2
    args.upstream_steps = steps
    args.handoff_positions = width
    built = []
    decoded = []

    class Method:
        def __init__(self, model, **kwargs):
            assert kwargs["temperature"] == 0
            assert kwargs["top_p"] == 1
            self.step = kwargs["latent_steps"]

        def build_latent_contexts(self, items):
            context = ((torch.full((1, 1, self.step + 20, 1), float(self.step)),),)
            built.append((items[0]["question"], self.step, context))
            return context, [[{"latent_steps": self.step, "input_ids": [1, 2, 3]}] * 3]

        def decode_with_context(self, items, **kwargs):
            context = kwargs["past_kv"]
            if self.step:
                assert context[0][0].shape[-2] == (width or self.step + 20)
                assert torch.all(context[0][0] == self.step)
            else:
                assert context is None
            decoded.append(
                (items[0]["question"], self.step, kwargs["receiver_mode"], context)
            )
            if context is not None:
                context[0][0].zero_()
            correct = kwargs["receiver_mode"] == "free" or self.step == 40
            return [
                {
                    "prediction": "1" if correct else "0",
                    "raw_prediction": "1",
                    "correct": correct,
                    "agents": [{"generated_tokens": 3, "input_ids": [1, 2, 3, 4]}],
                }
            ]

    dataset = SimpleNamespace(
        load=lambda **kwargs: [
            {"question": "a", "gold": "1"},
            {"question": "b", "gold": "1"},
        ]
    )
    cache = SimpleNamespace(get_layer_path=lambda layer: tmp_path)
    tracker = None
    if tracked:
        from latentmas_reprop.infrastructure.tracking.mlflow_tracker import (
            MLflowTracker,
        )

        tracker = MLflowTracker(
            tracking_uri=f"sqlite:///{tmp_path / 'traces.db'}",
            artifact_location=str(tmp_path / "mlflow_artifacts"),
        )
    with patch(
        "latentmas_reprop.application.receiver_reasoning.runner.LatentMASMethod",
        Method,
    ):
        summary, records = ReceiverReasoningUseCase(
            dataset_port=dataset, cache_port=cache, tracker_port=tracker
        ).execute(None, args)
    if len(steps) == 1:
        assert set(summary["upstream_levels"]) == {"0", "5"}
        assert "substitution_signal" not in summary
        assert summary["execution"]["n_records_expected"] == 8
        assert summary["execution"]["n_records_completed"] == 8
        assert summary["runtime"]["latent_steps_total"] == 30
        assert len(built) == 2
        assert all(torch.all(context[0][0] == 5) for _, _, context in built)
        assert all(
            r["handoff_positions"] == 25 for r in records if r["upstream_latent_steps"]
        )
        return
    assert {r["handoff_positions"] for r in records if r["upstream_latent_steps"]} == {
        10
    }
    assert len([r for r in records if r["handoff_condition"] == "no_handoff"]) == 4
    assert all(
        r["receiver_max_new_tokens"] == 4096
        for r in records
        if r["receiver_mode"] == "free"
    )
    assert summary["runtime"]["latent_steps_total"] == 300
    assert summary["runtime"]["upstream_prompt_tokens_total"] == 36
    assert summary["runtime"]["receiver_prompt_tokens_total"] == 48
    assert summary["runtime"]["output_tokens_total"] == 36
    assert summary["execution"]["n_records_successful"] == 12
    assert summary["runtime"]["peak_vram_bytes"] == 0
    assert all(
        r["sample_peak_vram_bytes"] == r["receiver_peak_vram_bytes"] == 0
        for r in records
    )
    assert all(r["cache_bytes"] == 40 for r in records if r["upstream_latent_steps"])
    assert all(
        r["cache_dtype"] == "torch.float32"
        for r in records
        if r["upstream_latent_steps"]
    )
    assert summary["upstream_statistics"]["40"]["context_count"] == 2
    assert (
        summary["upstream_statistics"]["40"]["upstream_source_cache_bytes"]["mean"]
        == 240
    )
    assert (
        summary["paired_comparisons"]["40_answer_only_vs_10_answer_only"][
            "rescue_count"
        ]
        == 2
    )
    assert (
        summary["upstream_levels"]["10"]["cells"]["free"]["execution_diagnostics"][
            "receiver_prompt_tokens"
        ]["total"]
        == 8
    )
    assert len(built) == 4
    assert len(decoded) == len(records) == 12
    assert all(torch.all(context[0][0] == step) for _, step, context in built)
    assert summary["substitution_signal"] == 1
    assert (
        summary["upstream_levels"]["10"]["paired_correctness"]["free_only_correct"] == 2
    )
    assert summary["upstream_levels"]["40"]["cells"]["answer_only"][
        "prediction_frequencies"
    ] == {"1": 2}
    assert len(list(tmp_path.glob("*/sample_results.jsonl"))) == 1
    if tracked:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracker.tracking_uri)
        experiment = client.get_experiment_by_name(args.tracking_experiment_name)
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.metrics["latent_steps_total"] == 300
        assert runs[0].data.metrics["peak_vram_bytes"] == 0
        assert len(runs[0].data.metrics) < 80
        assert not any("execution_diagnostics" in k for k in runs[0].data.metrics)
        assert runs[0].data.params == {
            str(k): str(v) for k, v in records[0]["config"].items()
        }

        traces = client.search_traces(
            locations=[experiment.experiment_id], run_id=runs[0].info.run_id
        )
        assert len(traces) == 2
        for trace in traces:
            assert trace.info.tags["peak_vram_gib"] == "0.000"
            assert trace.info.tags["latent_steps_total"] == "150"
            assert len(trace.data.spans) == 9
            names = {span.name for span in trace.data.spans}
            assert {
                "build_upstream_10",
                "build_upstream_40",
                "decode_0_answer_only",
                "decode_0_free",
            } <= names
            assert len(trace.info.assessments) == 14
            assert trace.info.trace_metadata["mlflow.sourceRun"] == runs[0].info.run_id
            assert len(trace.data.spans[0].outputs["cells"]) == 6
            root = trace.data.spans[0]
            assert root.outputs["runtime"]["latent_steps_total"] == 150
            assert root.outputs["runtime"]["output_tokens_total"] == 18
            assert root.outputs["execution"]["n_records_completed"] == 6
            assert all(
                span.attributes["peak_vram_bytes"] == 0 for span in trace.data.spans
            )
            assert all(
                span.attributes["peak_vram_reserved_bytes"] == 0
                for span in trace.data.spans
            )
            for span in trace.data.spans:
                if span.name.startswith("decode_"):
                    assert span.outputs["receiver_generated_tokens_per_sec"] > 0


@pytest.mark.parametrize("steps", ["40,10", "10,10", "-1,40", "", "10,20,40", "a,40"])
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


@pytest.mark.parametrize("temperature", [0.0, 0.6])
def test_transformers_decoding_respects_zero_temperature(temperature):
    from latentmas_reprop.infrastructure.models.text_generation import (
        generate_text_batch,
    )

    class Model:
        def generate(self, **kwargs):
            assert kwargs["do_sample"] is (temperature > 0)
            if temperature == 0:
                assert "temperature" not in kwargs
                assert "top_p" not in kwargs
            else:
                assert kwargs["temperature"] == temperature
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3]]), past_key_values=None
            )

    tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: "answer")
    outputs, _, counts = generate_text_batch(
        Model(),
        tokenizer,
        torch.device("cpu"),
        torch.tensor([[1, 2]]),
        temperature=temperature,
    )
    assert outputs == ["answer"]
    assert counts == [1]
