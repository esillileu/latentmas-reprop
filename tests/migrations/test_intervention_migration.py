"""End-to-end migration, source preservation and run resolution."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmas_reprop.domain.services.prompts import (
    build_intervention_p1_messages,
    resolve_p1_variant_id,
)
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage.reader import ResearchDataReader
from latentmas_reprop.infrastructure.storage.tensor_store import FileStorage
from tools.migrations.intervention_continuation import migrate
from tools.migrations.intervention_continuation.checkpoints import equal_kv
from tools.migrations.intervention_continuation.files import (
    file_hash,
    manifest,
    rename_new,
)
from tools.migrations.intervention_continuation.tracking import find_runs


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "storage" / "experiments"
    monkeypatch.setenv("LATENTMAS_STORAGE_DIR", str(tmp_path / "storage"))
    identity = {
        "model": "Qwen/Qwen3-8B",
        "forward_validation": "process_rng_v1",
        "weights": "revision",
        "tokenizer": "token-revision",
        "chat_template": "template",
        "dtype": "torch.float32",
        "backend": "transformers",
        "realignment": False,
        "realignment_digest": None,
        "task": "gsm8k",
        "split": "test",
        "p1_variant": resolve_p1_variant_id(SimpleNamespace()),
        "p1_messages": build_intervention_p1_messages(
            "{question}", SimpleNamespace(model_name="Qwen/Qwen3-8B")
        ),
    }
    provenance = {
        "torch": "test",
        "transformers": "test",
        "device": "cuda",
        "attention": "test",
    }
    values = torch.arange(19 * 2, dtype=torch.float32).reshape(1, 1, 19, 2)
    links = {}
    for step in (10, 15):
        source = (
            root
            / "gsm8k"
            / "Qwen3-8B"
            / "latent_mas"
            / "sequential"
            / f"ls{step}_intervention_continuation"
        )
        shared = source / "samples" / "000000" / "shared"
        shared.mkdir(parents=True)
        config = {
            "model_name": identity["model"],
            "task": "gsm8k",
            "split": "test",
            "latent_steps": step,
            "state_identity": identity,
            "state_provenance": provenance,
        }
        (source / "config.json").write_text(json.dumps(config))
        (source / "summary.json").write_text('{"test": true}')
        meta = {
            "status": "complete",
            "sample_index": 0,
            "question": "question",
            "split": "test",
            "prompt": "prompt",
            "prompt_token_ids": [1, 2, 3, 4],
            "prompt_length": 4,
            "latent_steps": step,
            "files": ["kv.pt", "hidden.pt"],
        }
        (shared / "result.json").write_text(json.dumps(meta))
        torch.save(
            ((values[..., : 4 + step, :].clone(), values[..., : 4 + step, :].clone()),),
            shared / "kv.pt",
        )
        torch.save(
            torch.arange((4 + step) * 2).reshape(4 + step, 2), shared / "hidden.pt"
        )
        result = source / "samples" / "000000" / "ple"
        result.mkdir()
        (result / "result.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "sample_index": 0,
                    "condition": "ple",
                    "files": ["hidden.pt", "extra.txt"],
                    "generated_text": "answer",
                    "correct": True,
                    "trace_id": f"trace-{step}",
                }
            )
        )
        (result / "extra.txt").write_text("extra")
        torch.save(torch.ones(2), result / "hidden.pt")
        links[source.relative_to(root).as_posix()] = [f"run{step}"]
    monkeypatch.setattr(migrate, "find_runs", lambda uri, experiments: links)
    return root, identity, values


def test_dry_run_apply_reuse_and_reader(inputs, tmp_path):
    root, _identity, values = inputs
    original = manifest(root)
    report = migrate.migrate(root, "mock://tracking", False)
    assert report["mode"] == "dry-run"
    assert report["samples"][0]["status"] == "mergeable"
    assert manifest(root) == original
    report = migrate.migrate(root, "mock://tracking", True)
    assert report["samples"][0]["status"] == "migrated"
    assert all(
        manifest(root / rel)
        == {
            k[len(rel) + 1 :]: v for k, v in original.items() if k.startswith(rel + "/")
        }
        for rel in [x["source"] for x in report["sources"]]
    )
    state = root / report["sources"][0]["state_path"]
    store = FileStorage(PathResolver(tmp_path))
    kv, hidden, _ = store.assemble_shared_state(state, "000000", 15)
    assert equal_kv(kv, ((values, values),))
    assert hidden.shape[0] == 19
    reader = ResearchDataReader(
        path_resolver=PathResolver(tmp_path), tensor_store=store
    )
    loaded, _, _ = reader.load_intervention_state("run15", "000000")
    assert equal_kv(loaded, kv)
    run_dir = next((state / "runs").glob("*/migration.json"))
    info = json.loads(run_dir.read_text())
    result = info["config"]["result_paths"][0]
    _, record = store.load_intervention_result(state / result["path"])
    assert record["generated_text"] == "answer"
    assert record["trace_id"] == result["trace_id"]
    assert (state / result["path"] / "extra.txt").read_text() == "extra"
    second = migrate.migrate(root, "mock://tracking", True)
    assert second["samples"][0]["status"] == "reused"


def test_conflict_is_reported_and_source_preserved(inputs):
    root, _identity, _ = inputs
    report = migrate.migrate(root, "mock://tracking", True)
    state = root / report["sources"][0]["state_path"]
    (state / "samples" / "000000" / "shared" / "ls10" / "result.json").write_text("{}")
    report = migrate.migrate(root, "mock://tracking", True)
    assert all(item["status"] == "failed" for item in report["samples"])
    assert all(
        (root / item["source"] / "samples" / "000000" / "shared" / "kv.pt").is_file()
        for item in report["sources"]
    )


def test_missing_provenance_still_migrates(inputs):
    root, _, _ = inputs
    source = next(root.glob("*/*/*/*/ls10*"))
    (source / "config.json").unlink()
    for item in root.glob("*/*/*/*/ls*/samples/000000/shared/result.json"):
        metadata = json.loads(item.read_text())
        metadata.pop("question")
        metadata.pop("split")
        item.write_text(json.dumps(metadata))
    before = manifest(source)
    report = migrate.migrate(root, "mock://tracking", True)
    assert report["samples"][0]["status"] == "migrated"
    assert manifest(source) == before
    state = json.loads(
        (root / report["sources"][0]["state_path"] / "state.json").read_text()
    )
    assert state["identity"]["model_revision"] == "unknown"
    assert state["provenance"]["rng_validation"] == "unknown"


def test_read_only_mlflow_run_link(inputs, tmp_path, monkeypatch, make_mlflow_db):
    import mlflow
    from mlflow.tracking import MlflowClient

    root, _, _ = inputs
    uri = make_mlflow_db(tmp_path / "tracking.db")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    client = MlflowClient(uri)
    experiment = client.create_experiment(
        "latentmas/intervention/continuation",
        artifact_location=(tmp_path / "artifacts").as_uri(),
    )
    source = next(root.glob("*/*/*/*/ls15*"))
    relative = source.relative_to(root).as_posix()
    run = client.create_run(
        experiment, tags={"rel_dir": relative, "data_dir": str(source)}
    )
    span = client.start_trace(
        "migration-source", experiment_id=experiment, run_id=run.info.run_id
    )
    client.end_trace(span.trace_id, outputs={"answer": "test"})
    mlflow.flush_trace_async_logging()
    assert client.get_trace(span.trace_id).info.trace_id == span.trace_id
    result_file = source / "samples" / "000000" / "ple" / "result.json"
    result = json.loads(result_file.read_text())
    result["trace_id"] = span.trace_id
    result_file.write_text(json.dumps(result))
    before = client.get_run(run.info.run_id)
    before_db = file_hash(tmp_path / "tracking.db")
    monkeypatch.setattr(migrate, "find_runs", find_runs)
    report = migrate.migrate(root, uri, True)
    assert not [item for item in report["samples"] if item["status"] == "failed"]
    after = client.get_run(run.info.run_id)
    assert file_hash(tmp_path / "tracking.db") == before_db
    assert before.data.tags == after.data.tags
    assert before.info == after.info
    assert client.get_trace(span.trace_id).info.trace_id == span.trace_id
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(
        path_resolver=resolver, tensor_store=FileStorage(resolver)
    )
    _, hidden, _ = reader.load_intervention_state(run.info.run_id, "000000")
    assert hidden.shape[0] == 19
    link = json.loads(
        (
            root
            / "intervention_continuation"
            / "migration_runs"
            / f"{run.info.run_id}.json"
        ).read_text()
    )
    config = json.loads((root / link["run_path"] / "config.json").read_text())
    result = config["result_paths"][0]
    assert result["trace_id"] == span.trace_id
    result_hash = Path(result["path"]).parts[-2]
    _, restored = reader.load_intervention_result(
        run.info.run_id, "000000", 15, result_hash, "ple"
    )
    assert restored["trace_id"] == span.trace_id


def test_no_replace_even_for_empty_directory(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "file").write_text("value")
    with pytest.raises(FileExistsError):
        rename_new(source, destination)
    assert source.is_dir()
    assert not list(destination.iterdir())


def test_ambiguous_mlflow_run_is_not_redirected(inputs, monkeypatch):
    root, _, _ = inputs
    sources = sorted(root.glob("*/*/*/*/ls*"))
    links = {source.relative_to(root).as_posix(): ["same_run"] for source in sources}
    monkeypatch.setattr(migrate, "find_runs", lambda uri, experiments: links)
    report = migrate.migrate(root, "mock://tracking", True)
    assert report["samples"][0]["status"] == "migrated"
    assert any(
        "Ambiguous MLflow" in item.get("reason", "") for item in report["samples"]
    )
    assert not (
        root / "intervention_continuation" / "migration_runs" / "same_run.json"
    ).exists()


def test_conflicting_legacy_checkpoint_is_quarantined(inputs):
    root, _identity, _ = inputs
    source = next(root.glob("*/*/*/*/ls10*"))
    kv_file = source / "samples" / "000000" / "shared" / "kv.pt"
    kv = torch.load(kv_file, weights_only=False)
    kv[0][0][..., 0, 0] += 1
    torch.save(kv, kv_file)
    before = manifest(source)
    report = migrate.migrate(root, "mock://tracking", True)
    assert report["samples"][0]["status"] == "conflict"
    assert manifest(source) == before
    assert not (
        root / report["sources"][0]["state_path"] / "samples" / "000000"
    ).exists()
