"""Argument parsing definitions and validation for intervention and acquisition harnesses."""

import argparse
from typing import Any


def add_intervention_args(
    parser: argparse.ArgumentParser, defaults: dict[str, Any]
) -> None:
    """Add command-line arguments for the intervention harness."""
    parser.add_argument(
        "--intervention",
        action="store_true",
        default=defaults.get("intervention", False),
        help="Enable latent communication intervention experiment harness (own, cross, zero).",
    )
    parser.add_argument(
        "--intervention_conditions",
        type=str,
        default=defaults.get("intervention_conditions", "own,cross,zero"),
        help="Comma-separated conditions for intervention: 'own', 'cross', 'zero'.",
    )
    parser.add_argument(
        "--cross_policy",
        type=str,
        choices=["shift_1", "derangement"],
        default=defaults.get("cross_policy", "shift_1"),
        help="Deterministic pairing policy for cross-condition.",
    )
    parser.add_argument(
        "--zero_mode",
        type=str,
        choices=["none", "zeros"],
        default=defaults.get("zero_mode", "none"),
        help="Representation of zero context: 'none' (past_key_values=None) or 'zeros' (zeroed cache tensors).",
    )
    parser.add_argument(
        "--save_raw_cache",
        action="store_true",
        default=defaults.get("save_raw_cache", False),
        help="Persist raw latent KV cache tensors to .cache/latent_interventions/ locally.",
    )
    parser.add_argument(
        "--tracking_experiment_name",
        type=str,
        default=defaults.get("tracking_experiment_name", "latentmas_intervention"),
        help="Experiment name for MLflow tracking.",
    )


def add_acquisition_args(
    parser: argparse.ArgumentParser, defaults: dict[str, Any]
) -> None:
    """Add command-line arguments for the receiver acquisition experiment."""
    parser.add_argument(
        "--acquisition",
        action="store_true",
        default=defaults.get("acquisition", False),
        help="Run the secret-digit receiver acquisition experiment.",
    )
    parser.add_argument(
        "--carrier_modes",
        "--context_modes",
        dest="carrier_modes",
        type=str,
        default=defaults.get(
            "carrier_modes", defaults.get("context_modes", "full,latent_only")
        ),
    )
    parser.add_argument(
        "--acquisition_conditions",
        type=str,
        default=defaults.get("acquisition_conditions", "own,cross,drop"),
    )
    parser.add_argument(
        "--acquisition_cross_policy",
        type=str,
        default=defaults.get("acquisition_cross_policy", "different_digit_shift_v1"),
    )
    parser.add_argument(
        "--save_hidden_states",
        action="store_true",
        default=defaults.get("save_hidden_states", False),
    )
    parser.add_argument(
        "--probe_sender_latents",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("probe_sender_latents", False),
    )
    parser.add_argument(
        "--probe_prompt_templates",
        type=int,
        default=defaults.get("probe_prompt_templates", 20),
    )
    parser.add_argument(
        "--probe_folds", type=int, default=defaults.get("probe_folds", 5)
    )
    parser.add_argument(
        "--probe_permutations",
        type=int,
        default=defaults.get("probe_permutations", 5000),
    )
    parser.add_argument(
        "--probe_backend",
        choices=["auto", "torch", "sklearn"],
        default=defaults.get("probe_backend", "auto"),
    )
    parser.add_argument(
        "--probe_workers", type=int, default=defaults.get("probe_workers", 0)
    )
    parser.add_argument("--probe_c", type=float, default=defaults.get("probe_c", 1.0))
    parser.add_argument(
        "--probe_max_iter", type=int, default=defaults.get("probe_max_iter", 1000)
    )
    parser.add_argument(
        "--probe_tol", type=float, default=defaults.get("probe_tol", 1e-4)
    )
    parser.add_argument(
        "--probe_batch_size",
        type=int,
        default=defaults.get("probe_batch_size", 256),
    )
    parser.add_argument(
        "--save_latent_states",
        action=argparse.BooleanOptionalAction,
        default=defaults.get("save_latent_states", True),
    )


def validate_intervention_and_acquisition_args(
    parser: argparse.ArgumentParser, parsed: argparse.Namespace
) -> None:
    """Validate cross-argument dependencies for intervention and acquisition runs."""
    if isinstance(parsed.intervention_conditions, str):
        parsed.intervention_conditions = [
            c.strip() for c in parsed.intervention_conditions.split(",") if c.strip()
        ]

    def normalize_csv(name: str, allowed: set[str]) -> list[str]:
        raw = getattr(parsed, name)
        values = [value.strip() for value in raw.split(",") if value.strip()]
        if len(values) != len(set(values)):
            parser.error(f"--{name} contains duplicate values")
        unknown = set(values) - allowed
        if unknown or not values:
            parser.error(f"--{name} contains invalid values: {sorted(unknown)}")
        return values

    parsed.carrier_modes = normalize_csv(
        "carrier_modes",
        {
            "full",
            "prompt_only",
            "latent_only",
            "latent_only_compact_debug",
        },
    )
    parsed.context_modes = parsed.carrier_modes
    parsed.acquisition_conditions = normalize_csv(
        "acquisition_conditions",
        {"own", "cross", "drop", "drop_position_matched"},
    )
    if parsed.intervention and parsed.acquisition:
        parser.error("--intervention and --acquisition are mutually exclusive")
    if parsed.acquisition:
        if parsed.method != "latent_mas" or parsed.task != "secret_digit":
            parser.error(
                "--acquisition requires --method latent_mas --task secret_digit"
            )
        if parsed.use_vllm or parsed.latent_steps <= 0:
            parser.error("--acquisition requires transformers and --latent_steps > 0")
        if parsed.acquisition_cross_policy != "different_digit_shift_v1":
            parser.error("unsupported --acquisition_cross_policy")
        if parsed.tracking_experiment_name != "latentmas_receiver_acquisition":
            parser.error(
                "--acquisition requires --tracking_experiment_name "
                "latentmas_receiver_acquisition"
            )
        if "cross" in parsed.acquisition_conditions and parsed.max_samples < 2:
            parser.error("cross acquisition requires at least two samples")
        if parsed.probe_folds < 2 or parsed.probe_prompt_templates < parsed.probe_folds:
            parser.error("probe templates must be at least the number of folds")
        if parsed.probe_permutations < 0 or parsed.probe_workers < 0:
            parser.error("probe permutations and workers must be non-negative")
        if parsed.probe_batch_size <= 0:
            parser.error("probe batch size must be positive")
