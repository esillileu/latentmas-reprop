"""Migration preserves every continuation condition and its branch step."""

import json
import shutil
from pathlib import Path

from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage.reader import ResearchDataReader
from latentmas_reprop.infrastructure.storage.tensor_store import FileStorage
from tools.migrations.intervention_continuation import migrate

from .test_intervention_migration import inputs as inputs


def test_all_conditions_preserve_results(inputs, tmp_path):
    root, _, _ = inputs
    source = next(root.glob("*/*/*/*/ls15*"))
    sample = source / "samples" / "000000"
    for condition in ("pe", "plce", "plpe"):
        target = sample / condition
        shutil.copytree(sample / "ple", target)
        record = json.loads((target / "result.json").read_text())
        record.update(condition=condition, trace_id=f"trace-{condition}")
        (target / "result.json").write_text(json.dumps(record))
    report = migrate.migrate(root, "mock://tracking", True)
    assert all(item["status"] == "migrated" for item in report["samples"])
    link = json.loads(
        (
            root / "intervention_continuation" / "migration_runs" / "run15.json"
        ).read_text()
    )
    config = json.loads((root / link["run_path"] / "config.json").read_text())
    resolver = PathResolver(tmp_path)
    reader = ResearchDataReader(
        path_resolver=resolver, tensor_store=FileStorage(resolver)
    )
    for item in config["result_paths"]:
        condition = item["condition"]
        step = 0 if condition == "pe" else 15
        result_hash = Path(item["path"]).parts[-2]
        _, record = reader.load_intervention_result(
            "run15", "000000", step, result_hash, condition
        )
        assert record["generated_text"] == "answer"
        assert record["trace_id"] == (
            "trace-15" if condition == "ple" else f"trace-{condition}"
        )
    assert {item["condition"] for item in config["result_paths"]} == {
        "pe",
        "ple",
        "plce",
        "plpe",
    }
