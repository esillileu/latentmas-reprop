"""Reuse unexhausted results across token budgets without re-executing generation."""

import json
import shutil
from pathlib import Path
from typing import Any

from ...infrastructure.storage.tensor_store import (
    is_condition_complete,
    save_json_atomic,
)
from .identity import canonical_hash


def should_compute_condition(
    state: Path,
    sample: str,
    condition_step: int,
    condition: str,
    ident: dict[str, Any],
) -> bool:
    """Return True if condition result is missing and cannot be reused from prior budgets."""
    res_dir = state / "samples" / sample / "results" / f"ls{condition_step}"
    curr_dir = res_dir / canonical_hash(ident) / condition
    if is_condition_complete(curr_dir):
        return False
    return find_reusable_result(res_dir, ident) is None


def find_reusable_result(
    candidate_dir: Path,
    current_identity: dict[str, Any],
) -> tuple[dict[str, Any], Path] | None:
    """Find a complete, unexhausted prior result that matches all parameters except budget.

    Returns the prior record and path if reusable, otherwise None.
    """
    if not candidate_dir.exists():
        return None

    # Filter keys to match between identities (everything except max_new_tokens)
    match_keys = {
        "state",
        "step",
        "condition",
        "task",
        "gold",
        "seed",
        "temperature",
        "top_p",
        "rng_policy",
        "decoder",
    }

    current_budget = int(current_identity.get("max_new_tokens", 0))

    # candidate_dir is state / samples / sample / results / ls{step}
    for hash_dir in candidate_dir.iterdir():
        if not hash_dir.is_dir():
            continue
        cond_dir = hash_dir / current_identity["condition"]
        if not is_condition_complete(cond_dir):
            continue

        result_file = cond_dir / "result.json"
        if not result_file.is_file():
            continue

        try:
            record = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        prior_ident = record.get("identity", {})
        if not all(prior_ident.get(k) == current_identity.get(k) for k in match_keys):
            continue

        # Check condition-specific keys (control_signal, intervention_token_ids, etc.)
        cond = current_identity["condition"]
        if cond == "rqlce":
            if prior_ident.get("control_signal") != current_identity.get(
                "control_signal"
            ):
                continue
            if prior_ident.get("intervention_token_ids") != current_identity.get(
                "intervention_token_ids"
            ):
                continue
        elif cond in {"rqlrqe", "rqlre", "rqlqe", "rqrqe"}:
            if prior_ident.get("intervention_prompt") != current_identity.get(
                "intervention_prompt"
            ):
                continue
            if prior_ident.get("intervention_token_ids") != current_identity.get(
                "intervention_token_ids"
            ):
                continue
            extra_keys = ("plpe_prompt", "receiver_role", "prompt", "think")
            if not all(
                prior_ident.get(k) == current_identity.get(k) for k in extra_keys
            ):
                continue

        prior_budget = int(prior_ident.get("max_new_tokens", 0))
        # We can only reuse if prior budget was smaller (or equal) and did NOT exhaust
        if prior_budget <= current_budget:
            # If budget was exhausted, we cannot reuse it with a higher budget
            if (
                record.get("budget_exhausted") is True
                or record.get("finish_reason") == "length"
            ):
                continue
            return record, cond_dir

    return None


def copy_reusable_result(
    source_dir: Path,
    target_dir: Path,
    target_identity: dict[str, Any],
    source_record: dict[str, Any],
) -> dict[str, Any]:
    """Copy files from source_dir to target_dir and update identity to target_identity."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for file_name in source_record.get("files", ["hidden.pt"]):
        src_file = source_dir / file_name
        dst_file = target_dir / file_name
        if src_file.is_file():
            shutil.copy2(src_file, dst_file)

    new_record = dict(source_record)
    new_record["identity"] = target_identity
    new_record["status"] = "complete"
    save_json_atomic(target_dir / "result.json", new_record)
    return new_record


def get_or_create_result(
    use_case: Any,
    directory: Path,
    state: Path,
    sample: str,
    condition_step: int,
    ident: dict[str, Any],
) -> dict[str, Any] | None:
    """Load existing complete result or copy a reusable unexhausted result from a prior budget."""
    if is_condition_complete(directory):
        _, record = use_case.storage.load_intervention_result(directory, ident)
        return record

    res_dir = state / "samples" / sample / "results" / f"ls{condition_step}"
    reusable = find_reusable_result(res_dir, ident)
    if reusable is not None:
        prior_record, prior_dir = reusable
        return copy_reusable_result(prior_dir, directory, ident, prior_record)
    return None
