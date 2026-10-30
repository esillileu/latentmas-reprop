"""Publication on the research filesystem, including interrupted retries."""

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tools.migrations.intervention_continuation import migrate
from tools.migrations.intervention_continuation.files import (
    rename_new,
    require_safe_path,
)

from .test_intervention_migration import inputs as inputs


def test_atomic_publication_and_retry_on_storage_filesystem(monkeypatch):
    with tempfile.TemporaryDirectory(prefix=".publication-test-", dir=".") as name:
        root = Path(name).resolve()
        parent = root / "state" / "samples"
        parent.mkdir(parents=True)
        destination = parent / "000000"
        stage = root / "stage"
        stage.mkdir()
        (stage / "result.json").write_text('{"status":"complete"}')
        real_symlink = os.symlink

        def interrupt(src, dst, **kwargs):
            assert not destination.exists()
            assert (parent / src / "result.json").is_file()
            raise RuntimeError("interrupted before completion marker")

        monkeypatch.setattr(os, "symlink", interrupt)
        with pytest.raises(RuntimeError, match="interrupted"):
            rename_new(stage, destination)
        assert not destination.exists()
        stage.mkdir()
        (stage / "result.json").write_text('{"status":"complete"}')
        monkeypatch.setattr(os, "symlink", real_symlink)
        rename_new(stage, destination)
        assert destination.is_symlink()
        assert (destination / "result.json").read_text() == '{"status":"complete"}'
        require_safe_path(destination, root, allow_published_leaf=True)
        populated = [
            p
            for p in (root / "state" / ".migration-data" / "samples").glob(".000000-*")
            if any(p.iterdir())
        ]
        assert len(populated) == 1
        other = root / "other"
        other.mkdir()
        (other / "result.json").write_text("different")
        with pytest.raises(FileExistsError):
            rename_new(other, destination)
        assert (destination / "result.json").read_text() == '{"status":"complete"}'


def test_existing_empty_directory_is_never_replaced(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "file").write_text("value")
    destination = tmp_path / "state" / "samples" / "000000"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        rename_new(source, destination)
    assert not list(destination.iterdir())
    assert (source / "file").read_text() == "value"


def test_retry_after_interrupted_empty_reservation(tmp_path):
    destination = tmp_path / "state" / "samples" / "000000"
    destination.parent.mkdir(parents=True)
    data = tmp_path / "state" / ".migration-data" / "samples"
    (data / ".000000-interrupted").mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    (source / "result.json").write_text("complete")
    rename_new(source, destination)
    assert (destination / "result.json").read_text() == "complete"
    assert (data / ".000000-interrupted").is_dir()


def test_concurrent_publishers_never_replace_target():
    with tempfile.TemporaryDirectory(prefix=".publication-test-", dir=".") as name:
        root = Path(name)
        destination = root / "state" / "samples" / "000000"
        destination.parent.mkdir(parents=True)
        sources = [root / "a", root / "b"]
        for index, source in enumerate(sources):
            source.mkdir()
            (source / "result.json").write_text(str(index))

        def publish(source):
            try:
                rename_new(source, destination)
                return "published"
            except FileExistsError:
                return "exists"

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(publish, sources))
        assert sorted(outcomes) == ["exists", "published"]
        assert (destination / "result.json").read_text() in {"0", "1"}


def test_interrupted_sample_publication_resumes_with_new_tensor_files(
    inputs, monkeypatch
):
    root, _, _ = inputs
    original = os.symlink

    def interrupt(*args, **kwargs):
        raise OSError("interrupted before completion marker")

    monkeypatch.setattr(os, "symlink", interrupt)
    first = migrate.migrate(root, "mock://tracking", True)
    assert first["samples"][0]["status"] == "failed"
    state = root / first["sources"][0]["state_path"]
    assert not (state / "samples" / "000000").exists()
    data = state / ".migration-data" / "samples"
    assert len([p for p in data.glob(".000000-*") if any(p.iterdir())]) == 1
    monkeypatch.setattr(os, "symlink", original)
    second = migrate.migrate(root, "mock://tracking", True)
    assert second["samples"][0]["status"] == "migrated"
    assert (state / "samples" / "000000").is_symlink()
    assert len([p for p in data.glob(".000000-*") if any(p.iterdir())]) == 1
    third = migrate.migrate(root, "mock://tracking", True)
    assert third["samples"][0]["status"] == "reused"
