"""Latent space realignment service and caching."""

import contextlib
from typing import Any

import torch

from ...domain.ports.cache_port import CacheLayer
from ..cache.manager import ExecutionCacheManager


class RealignmentManager:
    """Manages computation, disk caching, and application of latent realignment matrices."""

    def __init__(
        self,
        model_name: str,
        cache_manager: ExecutionCacheManager,
        args: Any = None,
    ) -> None:
        self.model_name = model_name
        self.cache_manager = cache_manager
        self.args = args
        self._matrices: dict[int, tuple[torch.Tensor | None, torch.Tensor]] = {}
        self.pre_aligned: torch.Tensor | None = None

    def _target_norm(
        self, input_embeds: torch.nn.Module, device: torch.device
    ) -> torch.Tensor:
        """Mean input-embedding norm, matching the original FP32 reduction."""
        weight = input_embeds.weight.detach()
        pieces: list[torch.Tensor] = []
        for start in range(0, weight.shape[0], 2048):
            rows = weight[start : start + 2048].to(device=device, dtype=torch.float32)
            pieces.append(rows.norm(dim=1))
        return torch.cat(pieces).mean().detach()

    def _solve_realignment(
        self,
        input_embeds: torch.nn.Module,
        output_embeds: torch.nn.Module,
        device: torch.device,
    ) -> torch.Tensor:
        input_weight = input_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        output_weight = output_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        rhs = torch.matmul(output_weight.T, input_weight)
        return torch.linalg.solve(gram + reg, rhs)

    def build_matrix(
        self, model: torch.nn.Module, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Return the alignment matrix, or None when the transform is identity.

        Hidden-state normalization still uses the input embedding norm when
        realignment is disabled. The ridge solve is skipped in that case.
        """
        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        if input_embeds is None or not hasattr(input_embeds, "weight"):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        target_norm = self._target_norm(input_embeds, device)
        if not getattr(self.args, "latent_space_realign", False):
            return None, target_norm
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if output_embeds is None or not hasattr(output_embeds, "weight"):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        return self._solve_realignment(input_embeds, output_embeds, device), target_norm

    def ensure_matrix(
        self, model: torch.nn.Module, device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Retrieve cached matrix or compute and cache persistently."""
        key = id(model)
        info = self._matrices.get(key)
        target_device = torch.device(device)

        if info is not None:
            matrix, target_norm = info
            matrix, target_norm = _move_pair(matrix, target_norm, target_device)
            self._matrices[key] = (matrix, target_norm)
            return matrix, target_norm

        sanitized_model = self.model_name.replace("/", "_").replace("\\", "_")
        realign_flag = (
            "realigned"
            if getattr(self.args, "latent_space_realign", False)
            else "identity"
        )
        cache_filename = f"{sanitized_model}_{realign_flag}_realign.pt"

        loaded = _load_cached_matrix(self.cache_manager, cache_filename, target_device)
        if loaded is not None:
            self._matrices[key] = loaded
            return loaded

        matrix, target_norm = self.build_matrix(model, target_device)
        target_norm = target_norm.to(device=target_device, dtype=torch.float32)
        _save_cached_matrix(self.cache_manager, cache_filename, matrix, target_norm)
        self._matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def apply(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Apply realignment, or identity, then normalize to the embedding norm."""
        matrix, target_norm = self.ensure_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = hidden_fp32 if matrix is None else torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.pre_aligned = aligned.detach().clone()
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)


def _move_pair(
    matrix: torch.Tensor | None,
    target_norm: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    if matrix is not None and matrix.device != device:
        matrix = matrix.to(device)
    if target_norm.device != device:
        target_norm = target_norm.to(device)
    return matrix, target_norm


def _load_cached_matrix(
    cache_manager: ExecutionCacheManager,
    filename: str,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor] | None:
    if not cache_manager.exists(CacheLayer.MODELS_REALIGN, filename):
        return None
    try:
        cached = cache_manager.load_torch(
            CacheLayer.MODELS_REALIGN,
            filename,
            map_location=device,
        )
    except Exception:
        return None
    target_norm = cached["target_norm"].to(device)
    if cached.get("identity", False):
        return None, target_norm
    return cached["matrix"].to(device), target_norm


def _save_cached_matrix(
    cache_manager: ExecutionCacheManager,
    filename: str,
    matrix: torch.Tensor | None,
    target_norm: torch.Tensor,
) -> None:
    payload: dict[str, Any] = {"target_norm": target_norm.detach().cpu()}
    if matrix is None:
        payload["identity"] = True
    else:
        payload["matrix"] = matrix.detach().cpu()
    with contextlib.suppress(Exception):
        cache_manager.save_torch(CacheLayer.MODELS_REALIGN, filename, payload)
