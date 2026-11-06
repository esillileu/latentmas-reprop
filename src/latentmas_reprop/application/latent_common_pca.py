"""Per-item common PCA over alignment and thinking trajectories, for any task."""

import pickle

import torch

from ..infrastructure.storage import load_json
from .latent_alignment_analysis import alignment_protocol_mismatches
from .latent_state_features import fit_pca, hidden_features

TRAJECTORY_TYPES = ("without_think", "with_think")
TENSOR_ERRORS = (
    OSError,
    ValueError,
    RuntimeError,
    TypeError,
    KeyError,
    AttributeError,
    EOFError,
    pickle.UnpicklingError,
)


def common_projection(trajectories, trajectory_weights=None):
    """Fit common axes with optional total weights per shared trajectory."""
    identities = [
        (
            t.get("benchmark"),
            t["source_run_id"],
            t["sample_id"],
            t["alignment"],
            t["trajectory_type"],
        )
        for t in trajectories
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate shared trajectory in PCA fit")
    hidden = torch.cat([t["hidden"] for t in trajectories])
    if trajectory_weights is None:
        trajectory_weights = [1.0] * len(trajectories)
    weights = torch.cat(
        [
            torch.full((len(t["hidden"]),), weight / len(t["hidden"]))
            for t, weight in zip(trajectories, trajectory_weights, strict=True)
        ]
    )
    basis = fit_pca(hidden, weights)
    rows = []
    for trajectory in trajectories:
        projected = (trajectory["hidden"].float() - basis["mean"]) @ basis[
            "components"
        ].T
        for step, coordinates in zip(trajectory["steps"], projected, strict=True):
            rows.append(
                {
                    key: trajectory[key]
                    for key in (
                        "source_run_id",
                        "sample_id",
                        "alignment",
                        "trajectory_type",
                    )
                }
                | (
                    {"benchmark": trajectory["benchmark"]}
                    if "benchmark" in trajectory
                    else {}
                )
                | {
                    "step": step,
                    "common_pca_1": coordinates[0].item(),
                    "common_pca_2": coordinates[1].item(),
                }
            )
    return basis, rows


def find_alignment_source(directory, metadata, other_directory=None):
    """Use an explicit source or a unique compatible opposite-alignment sibling."""
    if other_directory is not None:
        candidates = [other_directory.resolve()]
    else:
        candidates = [
            p.parent
            for p in sorted(directory.parent.glob("*/metadata.json"))
            if p.parent != directory
        ]
    compatible, notes = [], []
    for candidate in candidates:
        try:
            other = load_json(candidate / "metadata.json")
            mismatches = alignment_protocol_mismatches(metadata, other)
            if mismatches:
                if other_directory is not None:
                    notes.append(
                        f"Incompatible source {candidate}: {', '.join(mismatches)}"
                    )
            else:
                compatible.append((candidate, other))
        except (OSError, ValueError, KeyError, TypeError) as error:
            notes.append(f"Unavailable comparison source {candidate}: {error}")
    if len(compatible) == 1:
        return compatible, notes
    notes.append(
        "Opposite alignment unavailable"
        if not compatible
        else "Multiple compatible alignment sources; select --compare-directory"
    )
    return [], notes


def load_hidden_trajectory(directory, sample, name, metadata):
    """Read hidden/prompt tensors only; never fit from duplicated A/C or B/D rows."""
    root = directory / "samples" / f"{sample['index']:06d}" / "trajectories"
    states, errors = {}, []
    path = root / f"{name}.pt"
    if path.is_file():
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
            steps = data["steps"].tolist()
            if steps != list(range(1, metadata["latent_steps"] + 1)) or len(
                data["hidden"]
            ) != len(steps):
                raise ValueError("Hidden steps differ from recorded trajectory")
            hidden_features(data["hidden"])
            states.update(zip(steps, data["hidden"].float(), strict=True))
        except TENSOR_ERRORS as error:
            errors.append(f"Latent hidden: {error}")
    prompt_path = root / name / "prompt_hidden.pt"
    if prompt_path.is_file():
        try:
            prompt = torch.load(prompt_path, map_location="cpu", weights_only=True)
            origin = hidden_features(prompt.new_empty((0, prompt.shape[1])), prompt)
            if 0 in origin:
                if states and prompt.shape[1] != next(iter(states.values())).shape[0]:
                    raise ValueError("Prompt/latent hidden dimensions differ")
                states[0] = prompt[-1].float()
        except (*TENSOR_ERRORS, IndexError) as error:
            errors.append(f"Prompt hidden: {error}")
    steps = sorted(states)
    status = (
        "complete"
        if len(steps) == metadata["latent_steps"] + 1
        else "partial"
        if steps
        else "invalid"
        if errors
        else "missing"
    )
    entry = {
        "source_run_id": directory.name,
        "sample_id": sample["index"],
        "alignment": metadata["latent_space_realign"],
        "trajectory_type": name,
        "source_directory": str(directory),
        "status": status,
        "steps": steps,
        "errors": errors,
    }
    if steps:
        entry["hidden"] = torch.stack([states[k] for k in steps])
    return entry


def analyze_common_pca(directory, metadata, other_directory=None):
    """Keep a separate basis per item; missing groups stay explicit."""
    others, notes = find_alignment_source(directory, metadata, other_directory)
    sources = [(directory, metadata), *others]
    samples = {}
    for source, source_metadata in sources:
        for path in sorted((source / "samples").glob("*/sample.json")):
            sample = load_json(path)
            samples.setdefault(sample["index"], []).append(
                (source, source_metadata, sample)
            )
    results = []
    for index, entries in sorted(samples.items()):
        reference = entries[0][2]
        groups = {
            (alignment, name): {
                "alignment": alignment,
                "trajectory_type": name,
                "sample_id": index,
                "status": "missing",
                "steps": [],
                "errors": [],
            }
            for alignment in (False, True)
            for name in TRAJECTORY_TYPES
        }
        for source, source_metadata, sample in entries:
            for name in TRAJECTORY_TYPES:
                key = (source_metadata["latent_space_realign"], name)
                if any(
                    sample.get(field) != reference.get(field)
                    for field in ("question", "gold", "prompt_token_ids")
                ):
                    groups[key].update(
                        status="incompatible",
                        errors=["Question/gold/prompt IDs differ"],
                    )
                else:
                    groups[key] = load_hidden_trajectory(
                        source, sample, name, source_metadata
                    )
        available = [g for g in groups.values() if "hidden" in g]
        result = {
            "sample_id": index,
            "benchmark": metadata["task"],
            "model": metadata["model_name"],
            "status": "unavailable",
            "notes": notes,
            "groups": list(groups.values()),
            "fit_weighting": "Equal total weight per available shared trajectory; uniform within each trajectory",
        }
        if available:
            try:
                basis, coordinates = common_projection(available)
                result.update(
                    basis=basis,
                    coordinates=coordinates,
                    status="complete"
                    if all(g["status"] == "complete" for g in groups.values())
                    else "partial",
                )
            except (ValueError, RuntimeError) as error:
                result.update(status="invalid", error=str(error))
        results.append(result)
    return results
