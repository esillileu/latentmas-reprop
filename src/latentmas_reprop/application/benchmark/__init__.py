"""Benchmark run identity, stored results, and summary records."""

from .identity import build_run_id, config_snapshot, sample_key_for
from .report import build_metrics, runtime_metadata
from .store import BenchmarkRunStore, predictions_in_dataset_order

__all__ = [
    "BenchmarkRunStore",
    "build_metrics",
    "build_run_id",
    "config_snapshot",
    "predictions_in_dataset_order",
    "runtime_metadata",
    "sample_key_for",
]
