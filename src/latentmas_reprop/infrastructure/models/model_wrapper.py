from typing import Any

import torch

from ...domain.ports.model_port import LatentRolloutState, ModelPort, NextTokenState
from ..cache.manager import ExecutionCacheManager, get_cache_manager
from .chat_formatter import prepare_chat_batch
from .loader import init_vllm_backend, load_hf_causal_lm
from .realignment import RealignmentManager
from .rollout import generate_latent_hidden_state_batch, generate_latent_rollout
from .text_generation import forward_next_token_batch, generate_text_batch
from .vllm_service import generate_with_latent_embeddings, vllm_generate_text

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


class ModelWrapper(ModelPort):
    """Model adapter supporting HuggingFace Transformers and vLLM backends."""

    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        use_vllm: bool = False,
        args: Any = None,
        cache_manager: ExecutionCacheManager | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = (
            bool(getattr(args, "latent_space_realign", False)) if args else False
        )
        self.args = args
        self.cache_manager = cache_manager or get_cache_manager()
        self.realignment_manager = RealignmentManager(
            self.model_name, self.cache_manager, args
        )

        if self.use_vllm:
            (
                self.vllm_engine,
                self.tokenizer,
                self.HF_model,
                self.embedding_layer,
                self.HF_device,
            ) = init_vllm_backend(model_name, args)
            if self.HF_model is not None:
                self.realignment_manager.ensure_matrix(
                    self.HF_model, torch.device(self.HF_device)
                )
            return

        self.tokenizer, self.model = load_hf_causal_lm(model_name, self.device)
        if self.latent_space_realign:
            self.realignment_manager.ensure_matrix(self.model, self.device)

    @property
    def pre_aligned(self) -> torch.Tensor | None:
        return self.realignment_manager.pre_aligned

    @pre_aligned.setter
    def pre_aligned(self, val: torch.Tensor | None) -> None:
        self.realignment_manager.pre_aligned = val

    def _apply_latent_realignment(
        self, hidden: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        return self.realignment_manager.apply(hidden, model)

    def prepare_chat_batch(
        self,
        batch_messages: list[list[dict]],
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[list[str]]]:
        return prepare_chat_batch(
            self.tokenizer,
            self.device,
            batch_messages,
            add_generation_prompt,
            chat_template_kwargs,
        )

    def vllm_generate_text_batch(
        self,
        prompts: list[str],
        sampling_params: Any = None,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        return vllm_generate_text(
            self.vllm_engine,
            prompts,
            sampling_params=sampling_params,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate_with_latent_embeddings(
        self,
        judger_prompts: list[str],
        past_embedding: torch.Tensor,
        sampling_params: Any,
        vllm_device: torch.device | str,
    ) -> list[str]:
        return generate_with_latent_embeddings(
            self.vllm_engine,
            self.tokenizer,
            self.embedding_layer,
            judger_prompts,
            past_embedding,
            sampling_params,
            self.HF_device,
            vllm_device,
            self.model_name,
        )

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Any = None,
    ) -> tuple[list[str], Any]:
        return generate_text_batch(
            self.model,
            self.tokenizer,
            self.device,
            input_ids,
            attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            past_key_values=past_key_values,
        )

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(self.device)

    @torch.no_grad()
    def forward_next_token_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        output_hidden_states: bool = False,
        position_start: int | None = None,
    ) -> NextTokenState:
        return forward_next_token_batch(
            self.model,
            self.device,
            input_ids,
            attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            position_start=position_start,
        )

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> Any:
        return self._generate_latent_rollout(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        ).past_key_values

    @torch.no_grad()
    def generate_latent_batch_with_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentRolloutState:
        return self._generate_latent_rollout(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        )

    def _generate_latent_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentRolloutState:
        source_model = self.HF_model if hasattr(self, "HF_model") else self.model
        return generate_latent_rollout(
            self.model,
            self.device,
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            apply_realign_fn=self._apply_latent_realignment,
            source_model=source_model,
        )

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> tuple:
        return generate_latent_hidden_state_batch(
            self.HF_model,
            self.HF_device,
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            apply_realign_fn=self._apply_latent_realignment,
        )
