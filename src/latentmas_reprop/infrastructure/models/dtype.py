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


def is_bf16_hardware_supported(device: torch.device) -> bool:
    """Check if the CUDA device natively supports BF16 hardware acceleration (sm_80+)."""
    if not torch.cuda.is_available():
        return False
    if not torch.cuda.is_bf16_supported():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(index)
    return major >= 8


def resolve_model_dtype(device: torch.device) -> torch.dtype:
    """Use FP16 for model inference on every device, independent of BF16 support."""
    ensure_cuda_available(device)
    return torch.float16


def peak_allocated_bytes() -> int:
    """Max GPU memory allocated in this process. Zero when CUDA is absent."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())
