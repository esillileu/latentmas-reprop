"""Unified ResearchDataReader for loading artifacts and tensors across MLflow and TensorStore."""

import contextlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from ...domain.models.tensor_artifact import TensorManifest
from ..paths.resolver import PathResolver, get_path_resolver
from .tensor_store import TensorStore, get_tensor_store


class ResearchDataReader:
    """Unified reader interface for experiment artifacts, manifests, and heavy tensors.

    Abstracts away local filesystem layouts and MLflow artifact retrieval,
    allowing downstream analysis tools to query data by Run ID and Artifact ID
    without hardcoding paths.
    """

    def __init__(
        self,
        mlflow_client: Any | None = None,
        tensor_store: TensorStore | None = None,
        path_resolver: PathResolver | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.client = mlflow_client
        self.resolver = path_resolver or get_path_resolver()
        self.tensor_store = tensor_store or get_tensor_store()
        self.cache_dir = cache_dir or (self.resolver.cache_dir / "reader_scratch")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.provenance: list[dict[str, Any]] = []

    def load_artifact(
        self,
        run_id: str,
        artifact_path: str,
        allow_unscoped_fallback: bool = False,
    ) -> Any:
        """Download or read a small artifact (JSON, JSONL, YAML) from MLflow or local disk.

        To guarantee strict run isolation, only artifacts tied to run_id (via MLflow,
        run-scoped storage directories, or verified embedded run_id metadata)
        are loaded. Unscoped local files will not be loaded unless
        allow_unscoped_fallback=True is explicitly passed.
        """
        import logging
        logger = logging.getLogger(__name__)

        dest_dir = self.cache_dir / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = None
        if self.client:
            with contextlib.suppress(Exception):
                downloaded_path = Path(
                    self.client.download_artifacts(run_id, artifact_path, str(dest_dir))
                )

        # 1. Check strictly run-scoped local storage paths
        if downloaded_path is None or not downloaded_path.is_file():
            scoped_candidates = [
                self.resolver.storage_dir / run_id / artifact_path,
                self.resolver.cache_dir / "evaluation" / "runs" / run_id / artifact_path,
                self.resolver.artifacts_dir / run_id / artifact_path,
            ]
            for candidate in scoped_candidates:
                if candidate.is_file():
                    downloaded_path = candidate
                    break

        # 2. Check unscoped candidates with explicit safety verification
        if downloaded_path is None or not downloaded_path.is_file():
            unscoped_candidates = [
                self.resolver.artifacts_dir / artifact_path,
                self.resolver.cache_dir / artifact_path,
            ]
            for candidate in unscoped_candidates:
                if not candidate.is_file():
                    continue

                # Verify if file contents explicitly match run_id (for JSON)
                # Note: run_id represents concrete execution instance; config_id must NOT be used
                # to claim data provenance since multiple runs share the same config_id.
                is_matching = False
                if candidate.suffix == ".json":
                    try:
                        content = json.loads(candidate.read_text(encoding="utf-8"))
                        if isinstance(content, dict) and (
                            content.get("run_id") == run_id
                            or content.get("mlflow_run_id") == run_id
                        ):
                            is_matching = True
                    except Exception:
                        pass

                if is_matching:
                    downloaded_path = candidate
                    break
                elif allow_unscoped_fallback:
                    logger.warning(
                        "Loading unscoped artifact '%s' for run '%s' via allow_unscoped_fallback.",
                        candidate,
                        run_id,
                    )
                    downloaded_path = candidate
                    break

        if downloaded_path is None or not downloaded_path.is_file():
            raise FileNotFoundError(
                f"Artifact '{artifact_path}' not found for run '{run_id}' in MLflow or run-scoped local storage. "
                "Unscoped fallback was blocked to prevent cross-run pollution."
            )

        self.provenance.append({"run_id": run_id, "artifact_path": artifact_path, "local_path": str(downloaded_path)})

        if artifact_path.endswith(".jsonl"):
            return [
                json.loads(line)
                for line in downloaded_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif artifact_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(downloaded_path.read_text(encoding="utf-8"))
        elif artifact_path.endswith(".json"):
            return json.loads(downloaded_path.read_text(encoding="utf-8"))
        return downloaded_path.read_text(encoding="utf-8")

    def load_tensor(
        self,
        run_id: str,
        artifact_id: str,
        verify_checksum: bool = True,
        map_location: Any = None,
        allow_unscoped_fallback: bool = False,
    ) -> torch.Tensor:
        """Load a heavy tensor via TensorStore and manifest, checking existence and checksum.

        Unscoped legacy singletons (.cache/...) will not be loaded unless
        allow_unscoped_fallback=True is explicitly specified to avoid reading
        tensors from unrelated runs.
        """
        import logging
        logger = logging.getLogger(__name__)

        entry = self.tensor_store.get_entry(run_id, artifact_id)
        if entry is None and self.client:
            # Check if manifest can be downloaded from MLflow
            try:
                manifest_dict = self.load_artifact(run_id, "tensors/tensor_manifest.json")
                manifest = TensorManifest.from_dict(manifest_dict)
                entry = manifest.get_entry(artifact_id)
            except Exception:
                entry = None

        if entry is None:
            if allow_unscoped_fallback:
                # Legacy unscoped fallback: check .cache/evaluation/receiver_acquisition/sender_latent_states.pt etc.
                legacy_candidates = [
                    self.resolver.cache_dir / "evaluation" / "receiver_acquisition" / f"{artifact_id}.pt",
                    self.resolver.cache_dir / "models" / "realign" / f"{artifact_id}.pt",
                ]
                for candidate in legacy_candidates:
                    if candidate.is_file():
                        logger.warning(
                            "Loading unscoped tensor '%s' for run '%s' via allow_unscoped_fallback.",
                            candidate,
                            run_id,
                        )
                        return torch.load(candidate, map_location=map_location, weights_only=False)

            raise FileNotFoundError(
                f"Tensor artifact '{artifact_id}' not found in manifest or local store for run '{run_id}'. "
                "Unscoped fallback was blocked to prevent cross-run tensor pollution."
            )

        return self.tensor_store.load_tensor(
            entry,
            verify_checksum=verify_checksum,
            map_location=map_location,
        )

    def load_manifest(self, run_id: str) -> TensorManifest | None:
        manifest = self.tensor_store.get_manifest(run_id)
        if manifest:
            return manifest
        if self.client:
            try:
                manifest_dict = self.load_artifact(run_id, "tensors/tensor_manifest.json")
                return TensorManifest.from_dict(manifest_dict)
            except Exception:
                return None
        return None
