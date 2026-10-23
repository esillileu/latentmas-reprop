"""Resumable benchmark sample logs and finished summaries."""

import json
from dataclasses import fields
from typing import Any

from ...domain.models import BenchmarkMetrics
from ...domain.ports.cache_port import CacheLayer, CachePort
from ..common.reporter import write_json, write_jsonl
from .identity import sample_key_for


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
        """Load completed samples, dropping any corrupted or incomplete lines."""
        if not self.samples_path.is_file():
            return []
        lines = self.samples_path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        dirty = False
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dirty = True
                continue
            if not _is_valid_record(record):
                dirty = True
                continue
            records.append(record)
        if dirty:
            write_jsonl(self.samples_path, records)
        return records

    def has_keys(self, required_keys: set[str] | list[str]) -> bool:
        """Check if all required sample keys exist in valid completed records."""
        if not required_keys:
            return True
        return set(required_keys).issubset(self.completed_keys())

    def is_complete(self, target_samples: int | None = None) -> bool:
        records = self.read_records()
        if not records:
            return False
        if target_samples is not None:
            return len(records) >= target_samples and target_samples > 0
        summary = self.read_summary()
        if not summary or summary.get("status") != "complete":
            return False
        config = summary.get("config") or {}
        if config.get("requested_max_samples") != -1:
            return False
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
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.summary_path.with_suffix(".json.tmp")
        write_json(temporary, payload)
        temporary.replace(self.summary_path)

    def load_finished(
        self, target_samples: int | None = None
    ) -> tuple[BenchmarkMetrics, list[dict[str, Any]]]:
        summary = self.read_summary()
        records = self.read_records()
        if not records:
            raise FileNotFoundError(f"No records found for run {self.run_id}")
        if target_samples is not None and len(records) >= target_samples:
            records = records[:target_samples]
        known = {field.name for field in fields(BenchmarkMetrics)}
        if summary is not None:
            metrics_dict = dict(summary.get("metrics") or {})
            if target_samples is not None and len(records) != summary.get(
                "sample_count"
            ):
                correct = sum(1 for r in records if r.get("correct", False))
                metrics_dict["correct"] = correct
                metrics_dict["total"] = len(records)
                metrics_dict["max_samples"] = len(records)
                metrics_dict["accuracy"] = correct / max(1, len(records))
            metrics = BenchmarkMetrics(
                **{key: value for key, value in metrics_dict.items() if key in known}
            )
        else:
            correct = sum(1 for r in records if r.get("correct", False))
            metrics = BenchmarkMetrics(
                method="",
                model="",
                split="",
                seed=0,
                max_samples=len(records),
                accuracy=correct / max(1, len(records)),
                correct=correct,
                total=len(records),
            )
        predictions = [_without_sample_key(record) for record in records]
        return metrics, predictions


def _without_sample_key(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "sample_key"}


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
        predictions.append(_without_sample_key(record))
    return predictions


def _is_valid_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if not record.get("sample_key"):
        return False
    if record.get("prediction") is None:
        return False
    return "correct" in record
