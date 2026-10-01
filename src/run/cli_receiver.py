"""Receiver reasoning CLI options and validation."""

import argparse


def add_receiver_reasoning_args(parser: argparse.ArgumentParser, defaults: dict):
    parser.add_argument(
        "--receiver_reasoning",
        action="store_true",
        default=defaults.get("receiver_reasoning", False),
    )
    parser.add_argument(
        "--upstream_steps", default=defaults.get("upstream_steps", "10,40")
    )
    parser.add_argument(
        "--answer_only_max_new_tokens",
        type=int,
        default=defaults.get("answer_only_max_new_tokens", 64),
    )

    parser.add_argument(
        "--handoff_positions",
        type=int,
        default=defaults.get("handoff_positions"),
        help="Keep only the final N upstream KV positions; default transfers the full cache.",
    )


def validate_receiver_reasoning_args(parser, parsed):
    if not parsed.receiver_reasoning:
        return
    try:
        parsed.upstream_steps = [int(x) for x in str(parsed.upstream_steps).split(",")]
    except ValueError:
        parser.error("--upstream_steps requires comma-separated integers")
    if (
        not parsed.upstream_steps
        or any(step < 0 for step in parsed.upstream_steps)
        or parsed.upstream_steps != sorted(set(parsed.upstream_steps))
    ):
        parser.error("--upstream_steps requires increasing nonnegative values")
    # Zero is the automatically included no-handoff baseline, not a second build.
    parsed.upstream_steps = [step for step in parsed.upstream_steps if step > 0]
    if len(parsed.upstream_steps) not in (1, 2):
        parser.error("--upstream_steps requires 1-2 positive levels plus optional zero")
    if parsed.handoff_positions is not None and parsed.handoff_positions <= 0:
        parser.error("--handoff_positions must be positive")
    if parsed.method != "latent_mas" or parsed.use_vllm:
        parser.error("--receiver_reasoning requires latent_mas with transformers")
    if parsed.acquisition or parsed.intervention:
        parser.error("experiment use cases are mutually exclusive")
    if parsed.answer_only_max_new_tokens < 2 or parsed.max_new_tokens < 2:
        parser.error("receiver token limits must allow answer serialization")
    if parsed.max_samples == 0:
        parser.error("--max_samples must be positive or -1")
