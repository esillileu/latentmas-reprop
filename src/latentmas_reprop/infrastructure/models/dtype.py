"""Torch dtype selection and CUDA architecture checks."""

import torch


def dtype_name(dtype: torch.dtype) -> str:
    """Stable short name used in run identifiers and result metadata."""
    return str(dtype).removeprefix("torch.")


def ensure_cuda_architecture_supported(device: torch.device) -> None:
    """Fail before model load when the installed PyTorch build lacks this GPU."""
    if device.type != "cuda":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build has no available CUDA device. "
            f"torch={torch.__version__} torch.version.cuda={torch.version.cuda}."
        )
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    arch = f"sm_{major}{minor}"
    compiled = list(torch.cuda.get_arch_list())
    if arch not in compiled:
        name = torch.cuda.get_device_name(index)
        compiled_text = ", ".join(compiled) if compiled else "none"
        raise RuntimeError(
            f"Installed PyTorch ({torch.__version__}, CUDA {torch.version.cuda}) "
            f"does not include kernels for {name} ({arch}). "
            f"Compiled architectures: {compiled_text}."
        )


def resolve_model_dtype(device: torch.device) -> torch.dtype:
    """Select BF16 on supported GPUs, FP16 otherwise, and FP32 on CPU."""
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda":
        ensure_cuda_architecture_supported(device)
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def peak_allocated_bytes() -> int:
    """Max GPU memory allocated in this process. Zero when CUDA is absent."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.max_memory_allocated())
