"""Translate legacy use case names in paths."""

import re

BASE_STEM = r"ls\d+_seed-?\d+_temp[^_]+_top[^_]+_mt\d+(?:_split[^_]+)?_[A-Za-z0-9]+(?:_vllm)?(?:_think)?(?:_realign)?"
USE_CASE_TOKEN_RE = re.compile(
    rf"^(?P<base>{BASE_STEM})_(?P<token>single_intervention|intervention(?!(?:_kv|_continuation)(?:_|$))|acquisition)"
    r"(?P<rest>(?:_.*)?)$"
)
TRAJECTORY_RE = re.compile(
    rf"^(?P<head>{BASE_STEM}.*)_trajectory(?P<tail>(?:_smoke)?)$"
)
TOKEN_RENAMES = {
    "single_intervention": "intervention_continuation",
    "intervention": "intervention_kv",
    "acquisition": "representation_acquisition",
}


def new_params_stem(stem: str) -> str | None:
    """Return the renamed params directory name, or None if it is unaffected."""
    match = USE_CASE_TOKEN_RE.match(stem)
    if match:
        renamed = TOKEN_RENAMES[match["token"]]
        return f"{match['base']}_{renamed}{match['rest']}"
    match = TRAJECTORY_RE.match(stem)
    if match:
        return f"{match['head']}_representation_trajectory{match['tail']}"
    return None


def new_relpath(rel: str) -> str | None:
    """Rename ``<task>/<model>/<method>/<prompt>/<params>``; None if unaffected."""
    parts = rel.strip("/").split("/")
    if len(parts) != 5:
        return None
    renamed = new_params_stem(parts[-1])
    return None if renamed is None else "/".join([*parts[:-1], renamed])
