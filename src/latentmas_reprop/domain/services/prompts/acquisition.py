"""Prompt templates and builders for receiver acquisition experiments."""

SENDER_PROMPT_TEMPLATE_VERSION = "secret_digit_sender_v1"
RECEIVER_PROMPT_TEMPLATE_VERSION = "secret_digit_receiver_v2"
CROSS_PAIRING_POLICY = "different_digit_shift_v1"

SENDER_PROMPT_TEMPLATE = (
    "Memorize the secret digit {digit}. Communicate this digit through your internal "
    "state. Do not explain or output anything else."
)
RECEIVER_PROMPT = (
    "Identify the secret digit from the sender's internal state. "
    "Your answer must have exactly this form: The number is <digit>"
)
RECEIVER_ANSWER_PREFIX = "The number is "
CANDIDATE_DIGITS = tuple(str(i) for i in range(10))


def build_sender_messages(digit: int) -> list[dict[str, str]]:
    """Build sender message list for secret digit memorization."""
    return [{"role": "user", "content": SENDER_PROMPT_TEMPLATE.format(digit=digit)}]


def build_receiver_messages() -> list[dict[str, str]]:
    """Build receiver message list for digit identification."""
    return [{"role": "user", "content": RECEIVER_PROMPT}]
