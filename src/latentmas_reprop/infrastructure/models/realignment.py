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
        self._matrices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.pre_aligned: torch.Tensor | None = None

    def build_matrix(
        self, model: torch.nn.Module, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the ridge regression alignment matrix between output and input embeddings."""
        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )

        input_weight = input_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        output_weight = output_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if not getattr(self.args, "latent_space_realign", False):
            # Identity matrix fallback for normalization
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        return realign_matrix, target_norm

    def ensure_matrix(
        self, model: torch.nn.Module, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve cached matrix or compute and cache persistently."""
        key = id(model)
        info = self._matrices.get(key)
        target_device = torch.device(device)

        if info is not None:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)
                target_norm = target_norm.to(target_device)
                self._matrices[key] = (matrix, target_norm)
            return matrix, target_norm

        sanitized_model = self.model_name.replace("/", "_").replace("\\", "_")
        realign_flag = (
            "realigned"
            if getattr(self.args, "latent_space_realign", False)
            else "identity"
        )
        cache_filename = f"{sanitized_model}_{realign_flag}_realign.pt"

        if self.cache_manager.exists(CacheLayer.MODELS_REALIGN, cache_filename):
            try:
                cached_data = self.cache_manager.load_torch(
                    CacheLayer.MODELS_REALIGN,
                    cache_filename,
                    map_location=target_device,
                )
                matrix = cached_data["matrix"].to(target_device)
                target_norm = cached_data["target_norm"].to(target_device)
                self._matrices[key] = (matrix, target_norm)
                return matrix, target_norm
            except Exception:
                pass

        matrix, target_norm = self.build_matrix(model, target_device)
        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )

        with contextlib.suppress(Exception):
            self.cache_manager.save_torch(
                CacheLayer.MODELS_REALIGN,
                cache_filename,
                {"matrix": matrix.cpu(), "target_norm": target_norm.cpu()},
            )

        self._matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def apply(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Apply realignment transformation to hidden states."""
        matrix, target_norm = self.ensure_matrix(model, hidden.device)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        self.pre_aligned = aligned.detach().clone()
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)
