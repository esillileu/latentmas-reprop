"""Paired receiver decoding across two upstream latent compute levels."""

from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from transformers import set_seed

from ..domain.models import compute_sample_key
from ..domain.ports.cache_port import CacheLayer
from ..domain.services.kv_cache import clone_past_kv, get_past_kv_sequence_length
from ..domain.services.latent_mas import LatentMASMethod
from ..infrastructure.cache.manager import DEFAULT_CACHE_MANAGER
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import (
    build_run_stem,
    clean_config_args,
    write_json,
    write_jsonl,
    write_yaml,
)


def summarize_receiver_records(records: list[dict], steps: list[int]) -> dict:
    """Report descriptive aggregates without inferring causal effects."""
    levels = {}
    for step in steps:
        cells = {}
        paired = {}
        for mode in ("answer_only", "free"):
            rows = [
                r
                for r in records
                if r["upstream_latent_steps"] == step and r["receiver_mode"] == mode
            ]
            counts = Counter(r["prediction"] for r in rows)
            tokens = [r["receiver_generated_tokens"] for r in rows]
            cells[mode] = {
                "sample_count": len(rows),
                "accuracy": mean(r["correct"] for r in rows),
                "mean_generated_tokens": mean(tokens),
                "median_generated_tokens": median(tokens),
                "prediction_unique_count": len(counts),
                "prediction_frequencies": dict(counts.most_common()),
                "token_limit_count": sum(
                    r["receiver_token_limit_reached"] for r in rows
                ),
            }
            for row in rows:
                paired.setdefault(row["sample_id"], {})[mode] = row["correct"]
        transitions = Counter(
            "both_correct"
            if p["free"] and p["answer_only"]
            else "free_only_correct"
            if p["free"]
            else "answer_only_correct"
            if p["answer_only"]
            else "both_incorrect"
            for p in paired.values()
        )
        levels[str(step)] = {
            "cells": cells,
            "paired_correctness": {
                key: transitions[key]
                for key in (
                    "both_correct",
                    "free_only_correct",
                    "answer_only_correct",
                    "both_incorrect",
                )
            },
            "D": cells["free"]["accuracy"] - cells["answer_only"]["accuracy"],
        }
    return {
        "upstream_levels": levels,
        "substitution_signal": levels[str(steps[0])]["D"] - levels[str(steps[1])]["D"],
    }


class ReceiverReasoningUseCase:
    def __init__(
        self, dataset_port=None, cache_port=None, evaluator_port=None, tracker_port=None
    ):
        self.dataset_port = dataset_port or DEFAULT_DATASET_REGISTRY
        self.cache_port = cache_port or DEFAULT_CACHE_MANAGER
        self.evaluator_port = evaluator_port or DEFAULT_EVALUATOR
        self.tracker_port = tracker_port

    def execute(self, model: Any, args: Any) -> tuple[dict, list[dict]]:
        if args.method != "latent_mas" or args.use_vllm:
            raise ValueError("Receiver reasoning requires transformers LatentMAS")
        steps = args.upstream_steps
        if len(steps) != 2 or not 0 < steps[0] < steps[1]:
            raise ValueError("Expected two positive increasing upstream levels")
        items = list(self.dataset_port.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            items = items[: args.max_samples]
        if not items:
            raise ValueError("Receiver reasoning requires a nonempty dataset")
        config = clean_config_args(args)
        config["git_commit"] = get_git_commit_hash()
        stem = build_run_stem(
            "receiver_reasoning", args.task, args.model_name, len(items)
        )
        directory = (
            Path(self.cache_port.get_layer_path(CacheLayer.EVALUATION_RUNS)) / stem
        )
        records_path = directory / "sample_results.jsonl"
        summary_path = directory / "summary.json"
        config_path = directory / "resolved_config.yaml"
        write_yaml(config_path, config)
        tracker = self.tracker_port
        if tracker:
            tags = {
                "experiment_type": "receiver_reasoning",
                "git_commit": config["git_commit"],
            }
            if getattr(args, "version_tag", None):
                tags["version_tag"] = args.version_tag
            tracker.start_run(
                experiment_name=args.tracking_experiment_name,
                run_name=stem,
                tags=tags,
            )
            tracker.log_params(config)
        records = []
        try:
            for index, item in enumerate(items):
                sample_id = compute_sample_key(args.task, args.split, item["question"])
                for step in steps:
                    method = LatentMASMethod(
                        model,
                        latent_steps=step,
                        judger_max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        generate_bs=1,
                        args=args,
                        evaluator=self.evaluator_port,
                    )
                    set_seed(args.seed)
                    context, traces = method.build_latent_contexts([item])
                    length = get_past_kv_sequence_length(context)
                    for mode in ("answer_only", "free"):
                        set_seed(args.seed)
                        limit = (
                            args.answer_only_max_new_tokens
                            if mode == "answer_only"
                            else args.max_new_tokens
                        )
                        result = method.decode_with_context(
                            [item],
                            past_kv=clone_past_kv(context),
                            initial_traces=traces,
                            receiver_mode=mode,
                            max_new_tokens=limit,
                        )[0]
                        tokens = result["agents"][-1]["generated_tokens"]
                        records.append(
                            {
                                "sample_id": sample_id,
                                "sample_index": index,
                                "question": item["question"],
                                "gold": item.get("gold", ""),
                                "upstream_latent_steps": step,
                                "receiver_mode": mode,
                                "context_id": f"{sample_id}:{step}",
                                "upstream_context_sequence_length": length,
                                "prediction": result["prediction"],
                                "raw_receiver_output": result["raw_prediction"],
                                "correct": result["correct"],
                                "error_msg": result.get("error_msg"),
                                "receiver_generated_tokens": tokens,
                                "receiver_token_limit_reached": tokens >= limit,
                                "model": args.model_name,
                                "seed": args.seed,
                                "config": config,
                                "agents": result["agents"],
                            }
                        )
                        write_jsonl(records_path, records)
                    del context
                print(
                    f"[Receiver reasoning] {index + 1}/{len(items)} samples", flush=True
                )
            summary = summarize_receiver_records(records, steps) | {"config": config}
            write_json(summary_path, summary)
            if tracker:
                metrics = {"substitution_signal": summary["substitution_signal"]}
                for step, level in summary["upstream_levels"].items():
                    metrics[f"steps_{step}_D"] = level["D"]
                    for mode, cell in level["cells"].items():
                        for name in (
                            "accuracy",
                            "mean_generated_tokens",
                            "median_generated_tokens",
                            "prediction_unique_count",
                            "token_limit_count",
                        ):
                            metrics[f"steps_{step}_{mode}_{name}"] = cell[name]
                tracker.log_metrics(metrics)
                for path in (records_path, summary_path, config_path):
                    tracker.log_artifact(path, artifact_path="results")
                tracker.end_run(status="FINISHED")
            print(f"Artifacts: {directory}", flush=True)
            return summary, records
        except Exception:
            if tracker:
                if records:
                    tracker.log_artifact(records_path, artifact_path="results")
                tracker.end_run(status="FAILED")
            raise
