"""Measure nested sample/inference scopes without losing the run memory peak."""

from contextlib import contextmanager
from time import perf_counter

import torch


class RuntimeMeasurements:
    """Own CUDA peak-counter resets for the Pilot's sequential inference loop."""

    def __init__(self, model):
        device = torch.device(getattr(model, "device", "cpu"))
        self.device = (
            device if device.type == "cuda" and torch.cuda.is_available() else None
        )
        self._active = []
        self._run_peak = {"peak_vram_bytes": 0, "peak_vram_reserved_bytes": 0}
        self._capture()

    def _capture(self):
        peak = {
            "peak_vram_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device
            else 0,
            "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(self.device)
            if self.device
            else 0,
        }
        for values in [self._run_peak, *self._active]:
            for name, value in peak.items():
                values[name] = max(values[name], value)
            if "vram_start_allocated_bytes" in values:
                values["incremental_peak_vram_bytes"] = (
                    values["peak_vram_bytes"] - values["vram_start_allocated_bytes"]
                )
                values["incremental_peak_vram_reserved_bytes"] = (
                    values["peak_vram_reserved_bytes"]
                    - values["vram_start_reserved_bytes"]
                )

    def run_peak(self):
        self._capture()
        return dict(self._run_peak)

    @contextmanager
    def measure(self, span):
        """Publish scoped peaks even on failure; fold nested scopes into their parents."""
        if self.device:
            torch.cuda.synchronize(self.device)
        self._capture()
        allocated = torch.cuda.memory_allocated(self.device) if self.device else 0
        reserved = torch.cuda.memory_reserved(self.device) if self.device else 0
        measurements = {
            "peak_vram_bytes": allocated,
            "peak_vram_reserved_bytes": reserved,
            "vram_start_allocated_bytes": allocated,
            "vram_start_reserved_bytes": reserved,
            "incremental_peak_vram_bytes": 0,
            "incremental_peak_vram_reserved_bytes": 0,
        }
        if self.device:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._active.append(measurements)
        started = perf_counter()
        try:
            yield measurements
        finally:
            if self.device:
                torch.cuda.synchronize(self.device)
            measurements["latency_sec"] = perf_counter() - started
            self._capture()
            self._active.pop()
            for name, value in measurements.items():
                span.set_attribute(name, value)
