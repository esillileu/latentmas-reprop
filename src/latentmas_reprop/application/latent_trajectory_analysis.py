"""Shared-state extraction and readout joins without model weights or KV loading."""

import pickle
from collections import Counter

import torch

from ..infrastructure.storage import load_json, save_json_atomic
from .latent_state_features import hidden_features, logit_features, trajectory_geometry


def trajectory_metrics(
    data, prompt_hidden=None, markers=None, tokenizer=None, latent_steps=None
):
    """Extract each component independently and retain unavailable steps as rows."""
    saved_steps = data.get("steps")
    if latent_steps is None:
        latent_steps = len(saved_steps) if saved_steps is not None else 0
    expected = list(range(1, latent_steps + 1))
    step_error = any(data.get(key) is not None for key in ("hidden", "logits")) and (
        not torch.is_tensor(saved_steps)
        or saved_steps.ndim != 1
        or saved_steps.tolist() != expected
    )
    rows = {
        step: {
            "step": step,
            "hidden_status": "missing",
            "logits_status": "missing",
            "hidden_error": None,
            "logits_error": None,
            "logit_context": "shared_trajectory_raw_before_readout_prefill",
            "geometry_status": "missing",
            **dict.fromkeys(
                [
                    "l2_norm",
                    "distance_0",
                    "cosine_0",
                    "adjacent_l2",
                    "adjacent_cosine",
                    "direction_cosine",
                    "entropy",
                    "js_divergence",
                    "kl_current_previous",
                    "js_from_step1",
                    "think_logit",
                    "endthink_logit",
                    "think_probability",
                    "endthink_probability",
                    "top_token_ids",
                    "top_probabilities",
                    "top_tokens",
                    "pca_1",
                    "pca_2",
                    "effective_rank",
                ]
            ),
            "injected_vector_status": "requires_model_weights",
            "answer_scoring_status": "requires_conditional_scoring",
        }
        for step in range(latent_steps + 1)
    }
    hidden = data.get("hidden")
    if hidden is not None:
        try:
            if (
                step_error
                or not torch.is_tensor(hidden)
                or hidden.ndim != 2
                or len(hidden) != latent_steps
            ):
                raise ValueError("Hidden steps do not match the recorded trajectory")
            features = hidden_features(hidden)
            for step, values in features.items():
                rows[step].update(values, hidden_status="complete")
        except (ValueError, TypeError, RuntimeError) as error:
            for step in expected:
                rows[step].update(hidden_status="invalid", hidden_error=str(error))
    if prompt_hidden is not None:
        try:
            # Prompt origin can be analyzed even if the latent artifact is absent.
            if (
                not torch.is_tensor(prompt_hidden)
                or prompt_hidden.ndim != 2
                or not len(prompt_hidden)
            ):
                raise ValueError("Invalid prompt hidden shape")
            empty = prompt_hidden.new_empty((0, prompt_hidden.shape[1]))
            origin = hidden_features(empty, prompt_hidden)[0]
            rows[0].update(origin, hidden_status="complete")
            if hidden is not None and all(
                rows[k]["hidden_status"] == "complete" for k in expected
            ):
                for step, values in hidden_features(hidden, prompt_hidden).items():
                    rows[step].update(values)
        except (ValueError, TypeError, RuntimeError) as error:
            rows[0].update(hidden_status="invalid", hidden_error=str(error))
    valid_hidden = [k for k in rows if rows[k]["hidden_status"] == "complete"]
    if valid_hidden:
        states = [prompt_hidden[-1] if k == 0 else hidden[k - 1] for k in valid_hidden]
        try:
            coordinates, geometry = trajectory_geometry(
                torch.stack(states), valid_hidden
            )
            for coordinate in coordinates:
                rows[coordinate["step"]].update(
                    coordinate, **geometry, geometry_status="complete"
                )
        except (ValueError, RuntimeError) as error:
            for step in valid_hidden:
                rows[step].update(geometry_status="invalid", geometry_error=str(error))
    logits = data.get("logits")
    if logits is not None:
        try:
            if (
                step_error
                or not torch.is_tensor(logits)
                or logits.ndim != 2
                or len(logits) != latent_steps
            ):
                raise ValueError("Logit steps do not match the recorded trajectory")
            for step, values in logit_features(logits, markers, tokenizer).items():
                rows[step].update(values, logits_status="complete")
        except (ValueError, TypeError, RuntimeError) as error:
            for step in expected:
                rows[step].update(logits_status="invalid", logits_error=str(error))
    rows[0]["logits_error"] = "Step 0 logits not saved; LM-head weights required"
    return list(rows.values())


def analyze_trajectories(directory, output, tokenizer=None):
    """Load each shared trajectory once and preserve all source tensor files."""
    metadata = load_json(directory / "metadata.json")
    statuses, features = [], []
    for path in sorted((directory / "samples").glob("*/sample.json")):
        sample = load_json(path)
        for name in ("without_think", "with_think"):
            root = path.parent / "trajectories"
            data, prompt, errors = {}, None, {}
            for component, tensor_path in (
                ("trajectory", root / f"{name}.pt"),
                ("prompt", root / name / "prompt_hidden.pt"),
            ):
                if tensor_path.is_file():
                    try:
                        value = torch.load(
                            tensor_path, map_location="cpu", weights_only=True
                        )
                        if component == "trajectory":
                            if not isinstance(value, dict):
                                raise ValueError(
                                    "Trajectory artifact must be a dictionary"
                                )
                            data = value
                        else:
                            prompt = value
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        EOFError,
                        pickle.UnpicklingError,
                    ) as error:
                        errors[component] = str(error)
            rows = trajectory_metrics(
                data,
                prompt,
                sample["thinking_token_ids"],
                tokenizer,
                metadata["latent_steps"],
            )
            identity = {
                "source_run_id": directory.name,
                "benchmark": metadata["task"],
                "model": metadata["model_name"],
                "split": metadata.get("split"),
                "alignment": metadata["latent_space_realign"],
                "sample_id": sample["index"],
                "trajectory_type": name,
            }
            for row in rows:
                row.update(identity)
                if "trajectory" in errors and row["step"]:
                    row.update(
                        hidden_status="invalid",
                        logits_status="invalid",
                        hidden_error=errors["trajectory"],
                        logits_error=errors["trajectory"],
                    )
                if "prompt" in errors and row["step"] == 0:
                    row.update(hidden_status="invalid", hidden_error=errors["prompt"])
            complete = sum(
                r["hidden_status"] == r["logits_status"] == "complete"
                for r in rows
                if r["step"]
            )
            status = (
                "complete"
                if complete == metadata["latent_steps"]
                else "partial"
                if any(
                    r["hidden_status"] == "complete" or r["logits_status"] == "complete"
                    for r in rows
                )
                else "invalid"
                if errors
                or any(
                    r["hidden_status"] == "invalid" or r["logits_status"] == "invalid"
                    for r in rows
                )
                else "missing"
            )
            statuses.append(
                {
                    **identity,
                    "status": status,
                    "hidden_counts": dict(Counter(r["hidden_status"] for r in rows)),
                    "logits_counts": dict(Counter(r["logits_status"] for r in rows)),
                    "errors": errors,
                }
            )
            features.extend(rows)
            save_json_atomic(output / path.parent.name / f"{name}.analysis.json", rows)
    save_json_atomic(output / "tensor_status.json", statuses)
    return statuses, features


def join_readout_features(readouts, latent):
    """Expand shared features onto readouts without treating them as independent states."""
    lookup = {
        (r["source_run_id"], r["sample_id"], r["trajectory_type"], r["step"]): r
        for r in latent
    }
    joined = []
    for readout in readouts:
        key = (
            readout["source_run_id"],
            readout["sample_id"],
            readout["trajectory_type"],
            readout["step"],
        )
        state = lookup.get(
            key, {"hidden_status": "missing", "logits_status": "missing"}
        )
        joined.append(
            {
                **readout,
                **{key: value for key, value in state.items() if key not in readout},
                "readout_start_logits_status": "requires_lm_head_after_closing_prefill"
                if readout["condition"] in "cd"
                else state["logits_status"],
            }
        )
    return joined
