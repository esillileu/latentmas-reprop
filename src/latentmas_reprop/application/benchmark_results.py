"""Run identifiers and resumable benchmark result files."""

import json
from dataclasses import fields
from typing import Any

import torch
import transformers

from ..domain.models import BenchmarkMetrics, compute_sample_key
from ..domain.ports.cache_port import CacheLayer, CachePort
from ..infrastructure.models.dtype import peak_allocated_bytes
from ..infrastructure.paths.resolver import get_path_resolver
from .evaluation_service import (
    OUTPUT_TOKEN_DEFINITION,
    WALL_TIME_DEFINITION,
    summarize_predictions,
)

_RUN_FIELDS = (
    "method",
    "model_name",
    "task",
    "prompt",
    "latent_steps",
    "seed",
    "max_new_tokens",
    "temperature",
    "top_p",
    "generate_bs",
    "max_samples",
    "think",
    "latent_space_realign",
    "use_vllm",
    "split",
)


def format_setting(value: Any) -> str:
    """Format a config value so equivalent numbers share one run id."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def build_run_id(args: Any, dtype_name: str) -> str:
    """Identify a run by method, model, task, sampling, and dtype.

    ``max_samples`` stays at the requested value, including ``-1``. A five-sample
    smoke run therefore cannot overwrite or resume a full-test run.
    """
    model = str(args.model_name).replace("/", "_").replace("\\", "_")
    return (
        f"{args.method}__{model}__{args.task}__{args.prompt}"
        f"__ls{int(args.latent_steps)}__seed{int(args.seed)}__{dtype_name}"
        f"__mt{int(args.max_new_tokens)}"
        f"_temp{format_setting(float(args.temperature))}"
        f"_top{format_setting(float(args.top_p))}"
        f"_bs{int(args.generate_bs)}_ms{int(args.max_samples)}"
        f"_think{format_setting(bool(getattr(args, 'think', False)))}"
        f"_realign{format_setting(bool(getattr(args, 'latent_space_realign', False)))}"
        f"_vllm{format_setting(bool(getattr(args, 'use_vllm', False)))}"
        f"_split{args.split}"
    )


def sample_key_for(args: Any, item: dict) -> str:
    """Stable sample id shared by a fresh run and a resumed run."""
    return compute_sample_key(args.task, args.split, item["question"])


class BenchmarkRunStore:
    """JSONL sample log plus a summary that is written only when a run finishes."""

    def __init__(self, cache: CachePort, run_id: str) -> None:
        self.cache = cache
        self.run_id = run_id
        self.root = cache.get_layer_path(CacheLayer.EVALUATION_RUNS) / run_id
        self.samples_path = self.root / "samples.jsonl"
        self.summary_path = self.root / "summary.json"
        self.progress_path = self.root / "progress.json"

    def read_summary(self) -> dict[str, Any] | None:
        if not self.summary_path.is_file():
            return None
        return json.loads(self.summary_path.read_text(encoding="utf-8"))

    def read_records(self) -> list[dict[str, Any]]:
        """Load completed samples, dropping a truncated final line after a crash."""
        if not self.samples_path.is_file():
            return []
        lines = self.samples_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if index == len(lines) - 1:
                    self.samples_path.write_text(
                        "".join(
                            f"{json.dumps(item, ensure_ascii=False)}\n"
                            for item in records
                        ),
                        encoding="utf-8",
                    )
                    break
                raise
            records.append(record)
        return records

    def is_complete(self) -> bool:
        summary = self.read_summary()
        if not summary or summary.get("status") != "complete":
            return False
        records = self.read_records()
        metrics = summary.get("metrics") or {}
        return (
            summary.get("sample_count") == len(records)
            and metrics.get("max_samples") == len(records)
            and len(records) > 0
        )

    def completed_keys(self) -> set[str]:
        return {str(record["sample_key"]) for record in self.read_records()}

    def append(self, record: dict[str, Any]) -> None:
        """Append one finished sample. An existing key is left unchanged."""
        key = str(record["sample_key"])
        if key in self.completed_keys():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self.samples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def read_progress(self) -> dict[str, Any]:
        if not self.progress_path.is_file():
            return {"eval_time_sec": 0.0, "resumed_sessions": 0}
        return json.loads(self.progress_path.read_text(encoding="utf-8"))

    def add_eval_time(self, seconds: float) -> float:
        progress = self.read_progress()
        total = float(progress.get("eval_time_sec", 0.0)) + float(seconds)
        sessions = int(progress.get("resumed_sessions", 0)) + 1
        self.root.mkdir(parents=True, exist_ok=True)
        self.progress_path.write_text(
            json.dumps({"eval_time_sec": total, "resumed_sessions": sessions}),
            encoding="utf-8",
        )
        return total

    def write_complete(self, payload: dict[str, Any]) -> None:
        """Persist the finished summary without touching samples already stored."""
        if self.is_complete():
            return
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.summary_path)

    def load_finished(self) -> tuple[BenchmarkMetrics, list[dict[str, Any]]]:
        summary = self.read_summary()
        if summary is None:
            raise FileNotFoundError(f"No summary for run {self.run_id}")
        known = {field.name for field in fields(BenchmarkMetrics)}
        metrics = BenchmarkMetrics(
            **{key: value for key, value in summary["metrics"].items() if key in known}
        )
        predictions = [
            {key: value for key, value in record.items() if key != "sample_key"}
            for record in self.read_records()
        ]
        return metrics, predictions


def config_snapshot(args: Any) -> dict[str, Any]:
    """JSON-safe copy of the settings that define a benchmark run."""
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {key: values.get(key) for key in _RUN_FIELDS if key in values}


def predictions_in_dataset_order(
    store: BenchmarkRunStore, args: Any, selected: list[dict]
) -> list[dict]:
    """Return stored samples in dataset order, without the storage key."""
    records = {record["sample_key"]: record for record in store.read_records()}
    predictions: list[dict] = []
    for item in selected:
        key = sample_key_for(args, item)
        record = records.get(key)
        if record is None:
            raise RuntimeError(f"Missing stored sample for {key}")
        predictions.append(
            {field: value for field, value in record.items() if field != "sample_key"}
        )
    return predictions


def build_metrics(
    args: Any,
    predictions: list[dict],
    *,
    run_id: str,
    dtype_name: str,
    eval_seconds: float,
    model_load_seconds: float,
) -> BenchmarkMetrics:
    """Aggregate one run. Eval time excludes model loading."""
    summary = summarize_predictions(predictions)
    total = int(summary["total"])
    per_sample = (eval_seconds / total) if total else 0.0
    return BenchmarkMetrics(
        method=args.method,
        model=args.model_name,
        split=args.split,
        seed=args.seed,
        max_samples=total,
        accuracy=float(summary["accuracy"]),
        correct=int(summary["correct"]),
        total_time_sec=round(eval_seconds, 4),
        time_per_sample_sec=round(per_sample, 4),
        total=total,
        output_tokens_total=int(summary["output_tokens_total"]),
        output_tokens_mean=round(float(summary["output_tokens_mean"]), 4),
        latent_steps_total=int(summary["latent_steps_total"]),
        latent_steps_mean=round(float(summary["latent_steps_mean"]), 4),
        model_load_time_sec=round(model_load_seconds, 4),
        eval_time_sec=round(eval_seconds, 4),
        peak_vram_bytes=peak_allocated_bytes(),
        dtype=dtype_name,
        prompt=args.prompt,
        latent_steps=int(args.latent_steps),
        run_id=run_id,
    )


def runtime_metadata(
    model: Any, dtype_name: str, eval_seconds: float
) -> dict[str, Any]:
    """Record the library, GPU, dtype, and metric definitions used for a run."""
    gpu_name = None
    capability = None
    if torch.cuda.is_available():
        device = getattr(model, "device", torch.device("cuda"))
        index = device.index if getattr(device, "index", None) is not None else 0
        gpu_name = torch.cuda.get_device_name(index)
        major, minor = torch.cuda.get_device_capability(index)
        capability = f"{major}.{minor}"
    hf_model = getattr(model, "model", None)
    config = getattr(hf_model, "config", None)
    return {
        "git_commit": get_path_resolver().get_git_commit_hash(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "gpu_name": gpu_name,
        "gpu_capability": capability,
        "dtype": dtype_name,
        "model_revision": getattr(config, "_commit_hash", None),
        "eval_time_sec": round(eval_seconds, 4),
        "model_load_time_sec": round(float(getattr(model, "load_time_sec", 0.0)), 4),
        "peak_vram_bytes": peak_allocated_bytes(),
        "output_token_definition": OUTPUT_TOKEN_DEFINITION,
        "wall_time_definition": WALL_TIME_DEFINITION,
        "zero_latent_steps_note": (
            "latent_steps=0 runs the non-judger prompts and then drops their KV "
            "cache before the judger. It does not match a run that simply skips "
            "latent iterations while keeping the cache."
        ),
    }
