"""Common artifact generation, serialization, and tracker reporting utilities."""

import json
import time
from pathlib import Path
from typing import Any

import yaml


def build_run_stem(prefix: str, task: str, model_name: str, n_samples: int) -> str:
    """Generate canonical run filename stem."""
    sanitized_model = model_name.replace("/", "_").replace("\\", "_")
    timestamp = int(time.time())
    return f"{prefix}_{task}_{sanitized_model}_s{n_samples}_{timestamp}"


def clean_config_args(args: Any) -> dict[str, Any]:
    """Extract JSON/YAML serializable primitive attributes from args namespace."""
    args_dict = vars(args) if hasattr(args, "__dict__") else dict(args)
    return {
        k: v
        for k, v in args_dict.items()
        if isinstance(v, (str, int, float, bool, list, dict, type(None)))
    }


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    """Write list of dicts to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def write_yaml(path: Path, data: Any) -> None:
    """Write data to a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
