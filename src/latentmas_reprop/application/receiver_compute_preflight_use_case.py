"""Paired receiver compute curves with a single trajectory per handoff."""

import tempfile
import traceback
from copy import copy
from pathlib import Path

from transformers import set_seed

from ..domain.models import compute_sample_key
from ..domain.services.kv_cache import move_past_kv
from ..domain.services.latent_mas import LatentMASMethod
from ..infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from ..infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from ..infrastructure.tracking.mlflow_tracker import get_git_commit_hash
from .common.reporter import clean_config_args, write_json, write_jsonl, write_yaml
from .receiver_compute_preflight.analysis import CONDITIONS, export
from .receiver_compute_preflight.inference import (
    evaluate_prefix,
    generate_receiver,
    length_matched_donors,
)
from .receiver_reasoning.inference import build_upstream
from .receiver_reasoning.runtime import RuntimeMeasurements


class ReceiverComputePreflightUseCase:
    def __init__(self, dataset_port=None, evaluator_port=None, tracker_port=None):
        self.dataset = dataset_port or DEFAULT_DATASET_REGISTRY
        self.evaluator = evaluator_port or DEFAULT_EVALUATOR
        self.tracker = tracker_port

    def execute(self, model, args):
        if self.tracker is None:
            raise ValueError("Receiver compute preflight requires MLflow tracking")
        if args.method != "latent_mas" or args.use_vllm or args.task != "gsm8k":
            raise ValueError("Preflight requires GSM8K transformers LatentMAS")
        if (
            args.handoff_positions is not None
            or getattr(args, "latent_only", False)
            or getattr(args, "sequential_info_only", False)
        ):
            raise ValueError("Preflight requires full KV handoff without truncation")
        args = copy(args)
        args.temperature, args.top_p = 0.0, 1.0
        items = list(self.dataset.load(task=args.task, split=args.split))
        if args.max_samples > 0:
            items = items[: args.max_samples]
        ids = [compute_sample_key(args.task, args.split, i["question"]) for i in items]
        if len(ids) < 2 or len(set(ids)) != len(ids):
            raise ValueError("Preflight requires at least two unique paired samples")
        config = clean_config_args(args) | {
            "sample_ids": ids,
            "model_dtype": getattr(model, "dtype_name", None),
            "git_commit": get_git_commit_hash(),
            "handoff_mode": "full",
            "receiver_mode": "free",
            "donor_policy": "length_sorted_rotation",
        }
        tracker = self.tracker
        tracker.start_run(
            experiment_name=args.tracking_experiment_name,
            tags={
                "experiment_type": "receiver_compute_preflight",
                "git_commit": config["git_commit"],
                "version_tag": args.version_tag or "receiver-compute-preflight",
            },
        )
        tracker.log_params(config)
        records = []
        runtime = RuntimeMeasurements(model)
        with tempfile.TemporaryDirectory(prefix="receiver-compute-") as tmp:
            directory = Path(tmp)
            write_yaml(directory / "resolved_config.yaml", config)
            try:
                # No-handoff is U-independent: generate once per sample and reuse across U.
                baseline_method = self._method(model, args, 0)
                baseline = [
                    self._trajectory(baseline_method, item, None, args)
                    for item in items
                ]
                for u in args.upstream_steps:
                    method = self._method(model, args, u)
                    contexts, traces, metadata = [], [], []
                    for sid, item in zip(ids, items, strict=True):
                        set_seed(args.seed)
                        context, trace, measurements = build_upstream(
                            method, item, u, None, f"{sid}:{u}", tracker, runtime
                        )
                        contexts.append(move_past_kv(context, "cpu"))
                        traces.append(trace[0])
                        metadata.append(measurements)
                    donors = length_matched_donors(
                        [m["handoff_positions"] for m in metadata]
                    )
                    for i, item in enumerate(items):
                        for condition in CONDITIONS:
                            donor = i if condition == "matched" else donors[i]
                            if condition == "no_handoff":
                                trajectory, prompt, verified = baseline[i]
                                donor_id, donor_meta, donor_trace = None, {}, []
                            else:
                                trajectory, prompt, verified = self._trajectory(
                                    method, item, contexts[donor], args
                                )
                                donor_id, donor_meta, donor_trace = (
                                    ids[donor],
                                    metadata[donor],
                                    traces[donor],
                                )
                            cap_reached = len(trajectory) >= args.max_new_tokens
                            for budget in args.receiver_budgets:
                                row = evaluate_prefix(
                                    model.tokenizer,
                                    self.evaluator,
                                    item,
                                    trajectory,
                                    budget,
                                )
                                row.update(
                                    sample_id=ids[i],
                                    sample_index=i,
                                    question=item["question"],
                                    gold=item.get("gold", ""),
                                    reference_solution=item.get("solution"),
                                    upstream_steps=u,
                                    handoff_condition=condition,
                                    donor_id=donor_id,
                                    donor_sequence_length=donor_meta.get(
                                        "handoff_positions", 0
                                    ),
                                    recipient_upstream_metadata=metadata[i],
                                    upstream_metadata=donor_meta,
                                    upstream_agents=donor_trace,
                                    context_id=f"{donor_id}:{u}" if donor_id else None,
                                    free_generated_tokens=len(trajectory),
                                    free_cap_reached=cap_reached,
                                    valid=not (budget == "free" and cap_reached),
                                    prefix_verified=verified,
                                    model=args.model_name,
                                    seed=args.seed,
                                    **prompt,
                                )
                                records.append(row)
                            write_jsonl(directory / "sample_results.jsonl", records)
                        print(
                            f"[compute preflight] U={u} sample={i + 1}/{len(items)}",
                            flush=True,
                        )
                    del contexts
                metrics = export(directory, records, config)
                tracker.log_metrics(
                    {
                        f"accuracy_u{r['upstream_steps']}_{r['handoff_condition']}_r{r['receiver_budget']}": r[
                            "accuracy"
                        ]
                        for r in metrics["curves"]
                        if r["accuracy"] is not None
                    }
                )
            except (Exception, KeyboardInterrupt):
                write_json(
                    directory / "failure.json", {"traceback": traceback.format_exc()}
                )
                for path in directory.iterdir():
                    tracker.log_artifact(path, artifact_path="results")
                tracker.flush_traces()
                tracker.end_run(status="FAILED")
                raise
            for path in directory.iterdir():
                tracker.log_artifact(path, artifact_path="results")
            tracker.flush_traces()
            tracker.end_run(status="FINISHED")
        return metrics, records

    def _method(self, model, args, steps):
        return LatentMASMethod(
            model,
            latent_steps=steps,
            args=args,
            evaluator=self.evaluator,
            temperature=0.0,
            top_p=1.0,
            generate_bs=1,
        )

    def _trajectory(self, method, item, context, args):
        set_seed(args.seed)
        ids, prompt = generate_receiver(
            method, item, context, args.max_new_tokens, True
        )
        verified = False
        if args.verify_prefix:
            for budget in args.receiver_budgets:
                if budget == "free":
                    continue
                set_seed(args.seed)
                capped, capped_prompt = generate_receiver(method, item, context, budget)
                if capped != ids[:budget] or capped_prompt != prompt:
                    raise ValueError(
                        f"Greedy prefix differs from capped generation at R={budget}"
                    )
                expected = evaluate_prefix(
                    method.model.tokenizer, self.evaluator, item, ids, budget
                )
                actual = evaluate_prefix(
                    method.model.tokenizer, self.evaluator, item, capped, budget
                )
                if expected != actual:
                    raise ValueError("Capped evaluation differs from prefix evaluation")
            verified = True
        return ids, prompt, verified
