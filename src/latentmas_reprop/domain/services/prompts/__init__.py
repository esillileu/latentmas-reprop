"""Domain services for generating benchmark and acquisition prompt messages."""

from .acquisition import (
    CANDIDATE_DIGITS,
    CROSS_PAIRING_POLICY,
    RECEIVER_ANSWER_PREFIX,
    RECEIVER_PROMPT,
    RECEIVER_PROMPT_TEMPLATE_VERSION,
    SENDER_PROMPT_TEMPLATE,
    SENDER_PROMPT_TEMPLATE_VERSION,
    build_receiver_messages,
    build_sender_messages,
)
from .hierarchical_latent import build_agent_message_hierarchical_latent_mas
from .hierarchical_text import build_agent_messages_hierarchical_text_mas
from .sequential_latent import build_agent_message_sequential_latent_mas
from .sequential_text import build_agent_messages_sequential_text_mas
from .single_agent import build_agent_messages_single_agent

__all__ = [
    "CANDIDATE_DIGITS",
    "CROSS_PAIRING_POLICY",
    "RECEIVER_ANSWER_PREFIX",
    "RECEIVER_PROMPT",
    "RECEIVER_PROMPT_TEMPLATE_VERSION",
    "SENDER_PROMPT_TEMPLATE",
    "SENDER_PROMPT_TEMPLATE_VERSION",
    "build_agent_message_hierarchical_latent_mas",
    "build_agent_message_sequential_latent_mas",
    "build_agent_messages_hierarchical_text_mas",
    "build_agent_messages_sequential_text_mas",
    "build_agent_messages_single_agent",
    "build_receiver_messages",
    "build_sender_messages",
]
