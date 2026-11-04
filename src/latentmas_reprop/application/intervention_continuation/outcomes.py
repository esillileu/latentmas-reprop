"""Condition outcome representation and loading from experiment runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ConditionOutcome:
    """Outcomes and correctness vector for a single experimental condition."""

    condition: str
    sample_ids: list[int | str]
    correct: list[bool]
    dataset: str | None = None
    split: str | None = None
    golds: list[Any] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as proportion of correct predictions."""
        return self.n_correct / self.n_samples if self.correct else 0.0

    @property
    def n_correct(self) -> int:
        """Calculate total number of correct predictions."""
        return sum(1 for c in self.correct if bool(c))

    @property
    def n_samples(self) -> int:
        """Calculate total number of samples."""
        return len(self.correct)

    @classmethod
    def from_run(cls, run_dir: str | Path) -> ConditionOutcome:
        """Create a ConditionOutcome from a run directory containing samples.jsonl."""
        run_path = Path(run_dir)
        samples_path = run_path / "samples.jsonl" if run_path.is_dir() else run_path
        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found at {samples_path}")

        records: list[dict[str, Any]] = []
        with open(samples_path, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSON on line {idx + 1}") from exc

        if not records:
            raise ValueError(f"No records found in {samples_path}")

        run_dir_path = run_path if run_path.is_dir() else run_path.parent
        ds, sp = _load_state_meta(run_dir_path)
        return cls.from_records(
            records, run_dir=run_dir_path, default_dataset=ds, default_split=sp
        )

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        run_dir: Path | None = None,
        default_dataset: str | None = None,
        default_split: str | None = None,
    ) -> ConditionOutcome:
        """Build ConditionOutcome from raw sample record dictionaries."""
        if not records:
            raise ValueError("Records list cannot be empty")

        items, conditions = _parse_records(records, default_dataset, default_split)
        if len(conditions) > 1:
            raise ValueError(f"Multiple conditions: {sorted(conditions)}")
        cond = next(iter(conditions))

        _check_unique_sample_ids(items)
        items.sort(
            key=lambda x: (0, int(x[0])) if str(x[0]).isdigit() else (1, str(x[0]))
        )

        s_ids = [x[0] for x in items]
        corr = [x[1] for x in items]
        golds = [x[3] for x in items]

        dsets = {x[4] for x in items if x[4] is not None}
        if len(dsets) > 1:
            raise ValueError(f"Inconsistent datasets: {sorted(dsets)}")
        splits = {x[5] for x in items if x[5] is not None}
        if len(splits) > 1:
            raise ValueError(f"Inconsistent splits: {sorted(splits)}")

        _check_missing_ids(s_ids)

        out = cls(
            condition=cond,
            sample_ids=s_ids,
            correct=corr,
            dataset=next(iter(dsets)) if dsets else default_dataset,
            split=next(iter(splits)) if splits else default_split,
            golds=golds,
        )
        if run_dir and (run_dir / "summary.json").exists():
            _validate_summary(out, run_dir / "summary.json")
        return out


def _load_state_meta(dir_path: Path) -> tuple[str | None, str | None]:
    state_file = None
    cfg_path = dir_path / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        if "state_path" in cfg:
            state_rel = Path(cfg["state_path"])
            # Check relative to repo root or storage/experiments or dir_path
            for base in [
                Path(),
                Path("storage/experiments"),
                dir_path.parent.parent.parent,
            ]:
                cand = base / state_rel / "state.json"
                if cand.exists():
                    state_file = cand
                    break
    if not state_file:
        cand = dir_path.parent.parent / "state.json"
        if cand.exists():
            state_file = cand

    if state_file and state_file.exists():
        s = json.loads(state_file.read_text(encoding="utf-8")).get("identity", {})
        return s.get("task") or s.get("dataset"), s.get("split")
    return None, None


def _parse_records(
    records: list[dict[str, Any]], d_dataset: str | None, d_split: str | None
) -> tuple[list[tuple[Any, bool, str, Any, str | None, str | None]], set[str]]:
    items = []
    conds = set()
    for idx, r in enumerate(records):
        if r.get("status") != "complete":
            raise ValueError(
                f"Incomplete sample at index {idx}: status={r.get('status')}"
            )
        sid = (
            r.get("sample_index")
            if r.get("sample_index") is not None
            else r.get("sample_id")
        )
        if sid is None:
            raise ValueError(f"Record at index {idx} lacks sample identifier")

        ident = r.get("identity") if isinstance(r.get("identity"), dict) else {}
        cond = r.get("condition") or r.get("condition_name") or ident.get("condition")
        if not cond:
            raise ValueError(f"Record at index {idx} lacks condition")
        conds.add(str(cond))

        if r.get("correct") is None:
            raise ValueError(f"Record at index {idx} lacks 'correct'")
        corr = bool(r["correct"])

        gold = r.get("gold") if r.get("gold") is not None else ident.get("gold")
        sample_ds = r.get("dataset") or r.get("task") or ident.get("task")
        if sample_ds is not None and d_dataset is not None and sample_ds != d_dataset:
            raise ValueError(
                f"Task/dataset conflict: record '{sample_ds}' vs state '{d_dataset}'"
            )
        sample_sp = r.get("split") or ident.get("split")
        if sample_sp is not None and d_split is not None and sample_sp != d_split:
            raise ValueError(
                f"Split conflict: record '{sample_sp}' vs state '{d_split}'"
            )

        ds = sample_ds or d_dataset
        sp = sample_sp or d_split
        items.append((sid, corr, str(cond), gold, ds, sp))
    return items, conds


def _check_unique_sample_ids(items: list[tuple]) -> None:
    all_ids = [x[0] for x in items]
    if len(all_ids) != len(set(all_ids)):
        dups = {x for x in all_ids if all_ids.count(x) > 1}
        raise ValueError(f"Duplicate sample IDs found: {sorted(dups, key=str)}")


def _check_missing_ids(ids: list[Any]) -> None:
    if ids and all(
        isinstance(x, int) or (isinstance(x, str) and x.isdigit()) for x in ids
    ):
        ints = [int(x) for x in ids]
        if len(ints) != (max(ints) - min(ints) + 1):
            missing = set(range(min(ints), max(ints) + 1)) - set(ints)
            raise ValueError(f"Missing sample IDs: {sorted(missing)[:10]}")


def _validate_summary(out: ConditionOutcome, summary_path: Path) -> None:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    tot = data.get("sample_count")
    if tot is None and isinstance(data.get("metrics"), dict):
        tot = data["metrics"].get("sample_count")
    if tot is not None and out.n_samples != int(tot):
        raise ValueError(f"Sample count mismatch: out={out.n_samples}, sum={tot}")

    stats = None
    if "condition_statistics" in data and out.condition in data["condition_statistics"]:
        stats = data["condition_statistics"][out.condition]
    elif "metrics" in data and f"{out.condition}_accuracy" in data["metrics"]:
        m = data["metrics"]
        stats = {
            "accuracy": m[f"{out.condition}_accuracy"],
            "correct": m.get(f"{out.condition}_correct"),
            "completed": m.get(f"{out.condition}_completed"),
        }
    if stats:
        comp = stats.get("completed")
        if comp is not None and out.n_samples != int(comp):
            raise ValueError(
                f"Completed count mismatch: out={out.n_samples}, sum={comp}"
            )
        acc = stats.get("accuracy")
        if acc is not None and abs(out.accuracy - float(acc)) > 1e-6:
            raise ValueError(f"Accuracy mismatch: out={out.accuracy}, sum={acc}")
        corr = stats.get("correct")
        if corr is not None and out.n_correct != int(corr):
            raise ValueError(f"Correct count mismatch: out={out.n_correct}, sum={corr}")
