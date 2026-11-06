"""One PCA basis per model across all saved items and benchmarks."""

from collections import Counter, defaultdict

from ..infrastructure.storage import load_json
from .latent_common_pca import (
    TRAJECTORY_TYPES,
    common_projection,
    load_hidden_trajectory,
)


def global_projection(trajectories):
    """Balance benchmark, item, trajectory, then states within each trajectory."""
    items = defaultdict(set)
    counts = Counter()
    for trajectory in trajectories:
        key = trajectory["benchmark"], trajectory["sample_id"]
        items[key[0]].add(key[1])
        counts[key] += 1
    weights = [
        1
        / (
            len(items)
            * len(items[t["benchmark"]])
            * counts[t["benchmark"], t["sample_id"]]
        )
        for t in trajectories
    ]
    return common_projection(trajectories, weights)


def analyze_global_pca(directory, model=None):
    """Use saved state as-is; no freshness checks or current-run selection."""
    models = defaultdict(list)
    for path in sorted(directory.glob("*/*/*/metadata.json")):
        metadata = load_json(path)
        if model is not None and metadata["model_name"] != model:
            continue
        for sample_path in sorted((path.parent / "samples").glob("*/sample.json")):
            sample = load_json(sample_path)
            for name in TRAJECTORY_TYPES:
                trajectory = load_hidden_trajectory(path.parent, sample, name, metadata)
                trajectory["benchmark"] = metadata["task"]
                models[metadata["model_name"]].append(trajectory)
    results = []
    for name, trajectories in sorted(models.items()):
        available = [t for t in trajectories if "hidden" in t]
        result = {
            "model": name,
            "status": "unavailable",
            "groups": trajectories,
            "fit_weighting": "Equal benchmarks, equal items within benchmark, equal saved trajectories within item, uniform states within trajectory",
        }
        if available:
            try:
                basis, coordinates = global_projection(available)
                result.update(
                    basis=basis,
                    coordinates=coordinates,
                    status="complete"
                    if len(available) == len(trajectories)
                    and all(t["status"] == "complete" for t in trajectories)
                    else "partial",
                )
            except (ValueError, RuntimeError) as error:
                result.update(status="invalid", error=str(error))
        results.append(result)
    return results
