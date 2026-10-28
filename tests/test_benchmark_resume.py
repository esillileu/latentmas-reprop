"""Result identity, resume, and token aggregation for benchmark runs."""

import argparse

import torch

from latentmas_reprop.application.benchmark import BenchmarkRunStore, build_run_id
from latentmas_reprop.application.benchmark.evaluation import summarize_predictions
from latentmas_reprop.application.benchmark_use_case import BenchmarkUseCase
from latentmas_reprop.domain.models import compute_sample_key
from latentmas_reprop.domain.services.latent_mas import LatentMASMethod
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.models.text_generation import count_new_token_ids
from latentmas_reprop.infrastructure.paths.resolver import PathResolver


def _args(**overrides):
    values = {
        "method": "latent_mas",
        "model_name": "Qwen/Qwen3-4B",
        "task": "gsm8k",
        "prompt": "sequential",
        "latent_steps": 20,
        "seed": 42,
        "max_new_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "generate_bs": 1,
        "max_samples": -1,
        "think": False,
        "latent_space_realign": False,
        "use_vllm": False,
        "split": "test",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_run_id_separates_steps_seed_and_dtype():
    base = build_run_id(_args(), "float16")
    assert base != build_run_id(_args(latent_steps=40), "float16")
    assert base != build_run_id(_args(seed=0), "float16")
    assert base != build_run_id(_args(), "bfloat16")
    # max_samples and generate_bs should share the same run_id for seamless resumption
    assert base == build_run_id(_args(max_samples=5), "float16")
    assert base == build_run_id(_args(generate_bs=4), "float16")


class _Dataset:
    def __init__(self, rows):
        self.rows = rows

    def load(self, task, split="test", **kwargs):
        return list(self.rows)


class _Method:
    def __init__(self):
        self.seen = []

    def run_batch(self, items):
        self.seen.extend(item["question"] for item in items)
        results = []
        for item in items:
            results.append(
                {
                    "question": item["question"],
                    "gold": item["gold"],
                    "solution": item["solution"],
                    "prediction": item["gold"],
                    "raw_prediction": "boxed",
                    "agents": [
                        {
                            "name": "Judger",
                            "role": "judger",
                            "output": "boxed",
                            "generated_tokens": 4,
                            "latent_steps": 20,
                        }
                    ],
                    "correct": True,
                    "generated_tokens": 4,
                    "latent_steps_executed": 20,
                }
            )
        return results


def test_completed_samples_resume_and_are_not_overwritten(tmp_path):
    rows = [
        {"question": "one plus one", "solution": "2", "gold": "2"},
        {"question": "two plus two", "solution": "4", "gold": "4"},
    ]
    cache = ExecutionCacheManager(PathResolver(tmp_path))
    method = _Method()
    use_case = BenchmarkUseCase(
        dataset_port=_Dataset(rows),
        cache_port=cache,
        evaluator_port=object(),
    )
    use_case.create_method = lambda model, args: method
    model = argparse.Namespace(
        dtype_name="float16", load_time_sec=1.5, device=torch.device("cpu")
    )
    run_id = build_run_id(_args(max_samples=-1, generate_bs=1), "float16")
    args = _args(max_samples=-1, generate_bs=1, run_id=run_id)

    store = BenchmarkRunStore(cache, run_id)
    store.append(
        {
            "sample_key": compute_sample_key("gsm8k", "test", "one plus one"),
            "question": "one plus one",
            "gold": "2",
            "solution": "2",
            "prediction": "kept",
            "raw_prediction": "kept",
            "agents": [{"generated_tokens": 4, "latent_steps": 20}],
            "correct": True,
            "generated_tokens": 4,
            "latent_steps_executed": 20,
        }
    )

    metrics, preds = use_case.execute(model, args)
    assert method.seen == ["two plus two"]
    assert [item["prediction"] for item in preds] == ["kept", "4"]
    assert metrics.output_tokens_total == 8
    assert metrics.latent_steps_total == 40
    assert metrics.output_tokens_mean == 4
    assert metrics.total == 2
    assert metrics.dtype == "float16"
    assert metrics.model_load_time_sec == 1.5
    assert metrics.eval_time_sec == metrics.total_time_sec

    method.seen.clear()
    again, again_preds = use_case.execute(model, args)
    assert method.seen == []
    assert [item["prediction"] for item in again_preds] == ["kept", "4"]
    assert again.run_id == metrics.run_id

    other = _args(latent_steps=10)
    other_method = _Method()
    use_case.create_method = lambda model, args: other_method
    use_case.execute(model, other)
    assert other_method.seen == ["one plus one", "two plus two"]
    assert build_run_id(other, "float16") != metrics.run_id


def test_token_summary_keeps_latent_steps_separate():
    summary = summarize_predictions(
        [
            {"correct": True, "generated_tokens": 5, "latent_steps_executed": 30},
            {
                "correct": False,
                "agents": [{"generated_tokens": 7, "latent_steps": 30}],
            },
        ]
    )
    assert summary["output_tokens_total"] == 12
    assert summary["output_tokens_mean"] == 6
    assert summary["latent_steps_total"] == 60
    assert summary["correct"] == 1
    assert summary["total"] == 2


def test_new_token_count_excludes_prompt_padding():
    ids = torch.tensor([3, 4, 0, 0])
    assert count_new_token_ids(ids, pad_token_id=0) == 2


def test_zero_latent_steps_do_not_pass_cache_to_judger():
    calls = {}

    class _Model:
        device = torch.device("cpu")

        def prepare_chat_batch(self, messages, add_generation_prompt=True):
            return ["prompt"], None, None, [["tok"]]

        def generate_latent_batch(self, ids, attention_mask=None, **kwargs):
            return ("cache",)

        def generate_text_batch(self, ids, mask, **kwargs):
            calls["past"] = kwargs.get("past_key_values")
            return ["\\boxed{1}"], None, [3]

    model = _Model()

    def _tokenize(text, **kwargs):
        return {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

    _tokenize.convert_ids_to_tokens = lambda ids: ["t"]
    model.tokenizer = _tokenize
    method = LatentMASMethod(
        model,
        latent_steps=0,
        judger_max_new_tokens=8,
        generate_bs=1,
        args=_args(latent_steps=0, task="gsm8k"),
    )
    result = method.run_batch([{"question": "q", "gold": "1", "solution": "1"}])
    assert calls["past"] is None
    assert result[0]["judger_received_latent_cache"] is False
    assert result[0]["generated_tokens"] == 3
    assert result[0]["latent_steps_executed"] == 0


def test_corrupted_and_incomplete_samples_are_dropped_and_rewritten(tmp_path):
    cache = ExecutionCacheManager(PathResolver(tmp_path))
    store = BenchmarkRunStore(cache, "test_corrupted_run")
    store.root.mkdir(parents=True, exist_ok=True)

    # Sample 0: valid
    s0 = store.resolver.get_condition_dir(store.root, 0, "benchmark", create=True)
    (s0 / "result.json").write_text(
        '{"sample_index": 0, "prediction": "ans1", "correct": true}',
        encoding="utf-8",
    )
    # Sample 1: broken json
    s1 = store.resolver.get_condition_dir(store.root, 1, "benchmark", create=True)
    (s1 / "result.json").write_text(
        '{"sample_index": 1, broken json...', encoding="utf-8"
    )
    # Sample 2: incomplete record (missing 'prediction' key)
    s2 = store.resolver.get_condition_dir(store.root, 2, "benchmark", create=True)
    (s2 / "result.json").write_text(
        '{"sample_index": 2, "correct": false}',
        encoding="utf-8",
    )
    # Sample 3: valid record with failed prediction (prediction is null)
    s3 = store.resolver.get_condition_dir(store.root, 3, "benchmark", create=True)
    (s3 / "result.json").write_text(
        '{"sample_index": 3, "prediction": null, "correct": false}',
        encoding="utf-8",
    )
    # Sample 4: valid
    s4 = store.resolver.get_condition_dir(store.root, 4, "benchmark", create=True)
    (s4 / "result.json").write_text(
        '{"sample_index": 4, "prediction": "ans4", "correct": true}',
        encoding="utf-8",
    )

    # read_records should ignore broken and incomplete records, but keep valid records with null prediction
    records = store.read_records()
    assert len(records) == 3
    assert [r["sample_index"] for r in records] == [0, 3, 4]
