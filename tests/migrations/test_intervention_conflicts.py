"""Sample-level conflicts."""

import json
import shutil

import torch

from tools.migrations.intervention_continuation import migrate

from .test_intervention_migration import inputs as inputs


def test_conflict_does_not_block_other_samples(inputs):
    root, _, _ = inputs
    sources = list(root.glob("*/*/*/*/ls*"))
    for source in sources:
        first = source / "samples" / "000000"
        second = source / "samples" / "000001"
        shutil.copytree(first, second)
        for path in (second / "shared" / "result.json", second / "ple" / "result.json"):
            metadata = json.loads(path.read_text())
            metadata["sample_index"] = 1
            path.write_text(json.dumps(metadata))
    original = sources[0] / "samples" / "000000" / "shared" / "kv.pt"
    cache = torch.load(original, weights_only=False)
    cache[0][0][..., 0, 0] += 1
    torch.save(cache, original)
    report = migrate.migrate(root, "mock://tracking", True)
    assert {item["sample"]: item["status"] for item in report["samples"]} == {
        "000000": "conflict",
        "000001": "migrated",
    }
    assert original.is_file()
    for source in report["sources"]:
        state = root / source["state_path"]
        assert (
            state / "samples" / "000001" / "shared" / "ls0" / "result.json"
        ).is_file()


def test_equal_prefill_merges_despite_different_recorded_tokens(inputs):
    root, _, _ = inputs
    source = next(root.glob("*/*/*/*/ls10*"))
    path = source / "samples" / "000000" / "shared" / "result.json"
    meta = json.loads(path.read_text())
    meta["prompt_token_ids"] = [4, 3, 2, 1]
    path.write_text(json.dumps(meta))
    report = migrate.migrate(root, "mock://tracking", False)
    assert report["counts"]["multiple_checkpoints"] == 1
    assert report["counts"]["mergeable"] == 1
