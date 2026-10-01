"""Validate cell identities, shared contexts and execution configuration."""

import json
from collections import defaultdict

MODES = ("answer_only", "free")
CONFIG_KEYS = (
    "model_name",
    "task",
    "split",
    "prompt",
    "seed",
    "temperature",
    "top_p",
    "think",
    "latent_space_realign",
    "latent_only",
    "sequential_info_only",
    "max_new_tokens",
    "answer_only_max_new_tokens",
    "upstream_steps",
    "method",
    "use_vllm",
    "use_second_HF_model",
    "device",
    "device2",
    "git_commit",
    "handoff_positions",
    "include_no_handoff",
    "handoff_mode",
)
REQUIRED_CONFIG = (
    *CONFIG_KEYS[:7],
    "max_new_tokens",
    "answer_only_max_new_tokens",
    "upstream_steps",
)


def inspect_records(records, config, params):
    steps = config.get("upstream_steps", [])
    if (
        not isinstance(steps, list)
        or len(steps) not in (1, 2)
        or any(step <= 0 for step in steps)
        or steps != sorted(set(steps))
    ):
        raise ValueError(
            "Resolved config must specify one or two increasing upstream_steps"
        )
    levels = dict(zip(steps, ("low", "high"), strict=False))
    if config.get("include_no_handoff") or any(
        r.get("upstream_latent_steps") == 0 for r in records
    ):
        levels[0] = "no_handoff"
    cells = {f"{level}_{mode}": [] for level in levels.values() for mode in MODES}
    problems, issues = {key: [] for key in cells}, []

    def issue(text, affected=tuple(cells)):
        issues.append(text)
        for key in affected:
            problems[key].append(text)

    for key in REQUIRED_CONFIG:
        if key not in config:
            issue(f"Missing resolved config field: {key}")
    for key in CONFIG_KEYS:
        if key in params and key in config and str(params[key]) != str(config[key]):
            issue(f"MLflow parameter differs from resolved config: {key}")
    for row in records:
        step, mode = row.get("upstream_latent_steps"), row.get("receiver_mode")
        if step not in levels or mode not in MODES:
            issue(f"Unexpected cell: steps={step}, mode={mode}")
            continue
        key = f"{levels[step]}_{mode}"
        cells[key].append(row)
        if not row.get("sample_id"):
            issue(f"Missing sample_id in {key}", [key])
        for field in (
            "question",
            "gold",
            "prediction",
            "raw_receiver_output",
            "correct",
            "receiver_generated_tokens",
        ):
            if field not in row:
                issue(f"Missing {field} in {key}: {row.get('sample_id')}", [key])
        if not isinstance(row.get("correct"), bool):
            issue(f"Invalid correctness in {key}: {row.get('sample_id')}", [key])
        tokens = row.get("receiver_generated_tokens")
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
            issue(f"Invalid token count in {key}: {row.get('sample_id')}", [key])
        if "handoff_positions" in config:
            expected = (
                0
                if step == 0
                else row.get("upstream_source_sequence_length")
                if config["handoff_positions"] is None
                else config["handoff_positions"]
            )
            if (
                expected is None
                or row.get("handoff_positions") != expected
                or row.get("upstream_context_sequence_length") != expected
            ):
                issue(f"Invalid handoff width in {key}: {row.get('sample_id')}", [key])
        row_config = row.get("config", {})
        for field in REQUIRED_CONFIG:
            if field not in row_config:
                issue(f"Missing sample config {field} in {key}", [key])
        for field in CONFIG_KEYS:
            if row_config.get(field) != config.get(field):
                issue(
                    f"Sample config differs: {key}, {row.get('sample_id')}, {field}",
                    [key],
                )
        if row.get("model") != config.get("model_name") or row.get(
            "seed"
        ) != config.get("seed"):
            issue(f"Sample model/seed differs: {key}, {row.get('sample_id')}", [key])
    index_ids = defaultdict(set)
    for row in records:
        index = row.get("sample_index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            issue(f"Missing or invalid sample_index: {row.get('sample_id')}")
        else:
            index_ids[index].add(row.get("sample_id"))
    for index, ids in index_ids.items():
        if len(ids) > 1:
            issue(f"sample_index {index} refers to multiple samples")
    indexed = {}
    sets = {}
    for key, rows in cells.items():
        groups = defaultdict(list)
        for row in rows:
            groups[row.get("sample_id", "")].append(row)
        indexed[key] = {
            sid: group[0] for sid, group in groups.items() if len(group) == 1
        }
        sets[key] = set(groups)
        if not rows:
            issue(f"Empty cell: {key}", [key])
        for sid, group in groups.items():
            if len(group) > 1:
                issue(f"Duplicate sample in {key}: {sid} ({len(group)} rows)", [key])
    if len({frozenset(ids) for ids in sets.values()}) != 1:
        issues.append(
            "Cell sample sets differ; no intersection is used for paired metrics."
        )
    for sid in set().union(*sets.values()):
        entries = [(key, rows[sid]) for key, rows in indexed.items() if sid in rows]
        for field in ("question", "gold", "sample_index"):
            if (
                len({json.dumps(row.get(field), sort_keys=True) for _, row in entries})
                > 1
            ):
                issue(f"Sample {field} differs: {sid}", [key for key, _ in entries])
        for level in levels.values():
            keys = [f"{level}_{mode}" for mode in MODES]
            pair = [indexed[key].get(sid) for key in keys]
            if all(pair):
                for field in ("context_id", "upstream_context_sequence_length"):
                    if (
                        any(row.get(field) is None for row in pair)
                        or pair[0][field] != pair[1][field]
                    ):
                        issue(
                            f"Shared upstream context mismatch or missing {field}: {level}, {sid}",
                            keys,
                        )
    return (
        cells,
        indexed,
        sets,
        problems,
        {
            "issues": list(dict.fromkeys(issues)),
            "cell_sample_ids": {key: sorted(ids) for key, ids in sets.items()},
            "cell_problems": {
                key: list(dict.fromkeys(values)) for key, values in problems.items()
            },
            "all_sample_sets_equal": len({frozenset(ids) for ids in sets.values()})
            == 1,
        },
    )
