"""Read one saved shared KV at a time; summarize without mutating caches."""

import pickle

from ..domain.services.kv_cache import dense_kv_layers
from ..infrastructure.storage import load_json, load_tensor
from .latent_common_pca import TRAJECTORY_TYPES
from .latent_kv_features import kv_component_features


def analyze_kv(directory, samples=None, layers=None):
    metadata = load_json(directory / "metadata.json")
    statuses, summaries, positions = [], [], []
    for path in sorted((directory / "samples").glob("*/sample.json")):
        sample = load_json(path)
        if samples is not None and sample["index"] not in samples:
            continue
        for name in TRAJECTORY_TYPES:
            identity = {
                "source_run_id": directory.name,
                "benchmark": metadata["task"],
                "model": metadata["model_name"],
                "sample_id": sample["index"],
                "alignment": metadata["latent_space_realign"],
                "trajectory_type": name,
            }
            tensor_path = path.parent / "trajectories" / name / "kv.pt"
            status = {
                **identity,
                "status": "missing",
                "error": None,
                "path": str(tensor_path.resolve()),
            }
            if tensor_path.is_file():
                try:
                    data = load_tensor(tensor_path, map_location="cpu")
                    pairs, _ = dense_kv_layers(data["past_key_values"])
                    selected = (
                        list(range(len(pairs)))
                        if layers is None
                        else sorted(set(layers))
                    )
                    status.update(
                        status="complete",
                        layers=len(pairs),
                        prompt_length=data["prompt_length"],
                        latent_steps=data["latent_steps"],
                    )
                    for layer in selected:
                        if not 0 <= layer < len(pairs):
                            summaries.append(
                                {
                                    **identity,
                                    "layer": layer,
                                    "component": "KV",
                                    "status": "invalid",
                                    "error": "Layer outside saved cache",
                                }
                            )
                            status["status"] = "partial"
                            continue
                        for component, tensor in zip(
                            ("K", "V"), pairs[layer], strict=True
                        ):
                            key = {**identity, "layer": layer, "component": component}
                            try:
                                summary, rows = kv_component_features(
                                    tensor, data["prompt_length"], data["latent_steps"]
                                )
                                summaries.append({**key, **summary})
                                positions.extend({**key, **row} for row in rows)
                                if summary["status"] != "complete":
                                    status["status"] = "partial"
                            except (ValueError, RuntimeError, TypeError) as error:
                                summaries.append(
                                    {**key, "status": "invalid", "error": str(error)}
                                )
                                status["status"] = "partial"
                    del data, pairs
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    TypeError,
                    KeyError,
                    EOFError,
                    pickle.UnpicklingError,
                ) as error:
                    status.update(status="invalid", error=str(error))
            statuses.append(status)
    return statuses, summaries, positions
