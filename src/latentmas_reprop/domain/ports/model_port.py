from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class NextTokenState:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass(frozen=True)
class LatentRolloutState:
    """Detached last-layer states captured at each sender latent step."""

    past_key_values: Any
    hidden_pre_realign: torch.Tensor
    latent_post_realign: torch.Tensor


class ModelPort(ABC):
    """Port interface for LLM inference, embedding, and latent state operations."""

    @abstractmethod
    def prepare_chat_batch(
        self,
        batch_messages: list[list[dict]],
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[list[str]]]:
        """Format and tokenize a batch of chat messages."""
        raise NotImplementedError

    @abstractmethod
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
        """Generate text completions for a tokenized batch."""
        raise NotImplementedError

    @abstractmethod
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> Any:
        """Forward through the model recurrently in latent space without text generation."""
        raise NotImplementedError

    @abstractmethod
    def generate_latent_batch_with_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentRolloutState:
        """Generate latent KV and return detached pre/post-realignment step states."""
        raise NotImplementedError

    @abstractmethod
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize arbitrary string text into tensor IDs."""
        raise NotImplementedError

    @abstractmethod
    def forward_next_token_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        output_hidden_states: bool = False,
        position_start: int | None = None,
    ) -> NextTokenState:
        """Return next-token logits without generation or a new cache."""
        raise NotImplementedError
