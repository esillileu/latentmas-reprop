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
    ) -> Any:
        """Download or read an artifact (JSON, JSONL, YAML) from MLflow or run-scoped local storage.

        To guarantee strict run isolation, only artifacts tied to run_id (via MLflow
        or run-scoped storage directories) are loaded.
        """
        dest_dir = self.cache_dir / run_id
        dest_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = None
        if self.client:
            with contextlib.suppress(Exception):
                downloaded_path = Path(
                    self.client.download_artifacts(run_id, artifact_path, str(dest_dir))
                )

        # Check strictly run-scoped local storage paths
        if downloaded_path is None or not downloaded_path.is_file():
            scoped_candidates = [
                self.resolver.storage_dir / run_id / artifact_path,
                self.resolver.storage_dir
                / "evaluation"
                / "runs"
                / run_id
                / artifact_path,
                self.resolver.cache_dir
                / "evaluation"
                / "runs"
                / run_id
                / artifact_path,
                self.resolver.artifacts_dir / run_id / artifact_path,
            ]
            for candidate in scoped_candidates:
                if candidate.is_file():
                    downloaded_path = candidate
                    break

        if downloaded_path is None or not downloaded_path.is_file():
            raise FileNotFoundError(
                f"Artifact '{artifact_path}' not found for run '{run_id}' in MLflow or run-scoped local storage."
            )

        self.provenance.append(
            {
                "run_id": run_id,
                "artifact_path": artifact_path,
                "local_path": str(downloaded_path),
            }
        )

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
        sample_key: str = "",
        branch_id: str = "",
        segment_id: str = "",
    ) -> torch.Tensor:
        """Load a heavy tensor via TensorStore and manifest, checking existence and checksum."""
        entry = self.tensor_store.get_entry(
            run_id,
            artifact_id,
            sample_key=sample_key,
            branch_id=branch_id,
            segment_id=segment_id,
        )
        if entry is None and self.client:
            # Check if manifest can be downloaded from MLflow
            try:
                manifest_dict = self.load_artifact(
                    run_id, "tensors/tensor_manifest.json"
                )
                manifest = TensorManifest.from_dict(manifest_dict)
                entry = manifest.get_entry(
                    artifact_id,
                    sample_key=sample_key,
                    branch_id=branch_id,
                    segment_id=segment_id,
                )
            except Exception:
                entry = None

        if entry is None:
            raise FileNotFoundError(
                f"Tensor artifact '{artifact_id}' not found in manifest or local store for run '{run_id}'."
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
                manifest_dict = self.load_artifact(
                    run_id, "tensors/tensor_manifest.json"
                )
                return TensorManifest.from_dict(manifest_dict)
            except Exception:
                return None
        return None
