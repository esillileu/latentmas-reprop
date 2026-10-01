"""Torch dtype selection and CUDA availability checks."""

import torch


def dtype_name(dtype: torch.dtype) -> str:
    """Stable short name used in run identifiers and result metadata."""
    return str(dtype).removeprefix("torch.")


def ensure_cuda_available(device: torch.device) -> None:
    """Require CUDA availability; let PyTorch resolve kernel compatibility at runtime."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build has no available CUDA device. "
            f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}."
        )


def resolve_model_dtype(device: torch.device) -> torch.dtype:
    """Select BF16 on supported GPUs, FP16 otherwise, and FP32 on CPU."""
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda":
        ensure_cuda_available(device)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def peak_allocated_bytes() -> int:
    """Max GPU memory allocated in this process. Zero when CUDA is absent."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())
