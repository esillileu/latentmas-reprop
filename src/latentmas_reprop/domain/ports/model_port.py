from abc import ABC, abstractmethod
from typing import Any

import torch


class ModelPort(ABC):
    """Port interface for LLM inference, embedding, and latent state operations."""

    @abstractmethod
    def prepare_chat_batch(
        self,
        batch_messages: list[list[dict]],
        add_generation_prompt: bool = True,
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
    def tokenize_text(self, text: str) -> torch.Tensor:
        """Tokenize arbitrary string text into tensor IDs."""
        raise NotImplementedError
