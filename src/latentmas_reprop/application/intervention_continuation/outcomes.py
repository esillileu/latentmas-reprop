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
        if not self.correct:
            return 0.0
        return self.n_correct / self.n_samples

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
        """Create a ConditionOutcome from a run directory containing samples.jsonl.

        Validates:
        - Presence of samples.jsonl
        - Samples are sorted by sample_id / sample_index
        - No duplicate or missing sample IDs
        - All samples have complete status (no unfinished samples)
        - Matching accuracy against summary.json if present
        """
        run_path = Path(run_dir)
        samples_path = run_path / "samples.jsonl" if run_path.is_dir() else run_path

        if not samples_path.exists():
            raise FileNotFoundError(f"Samples file not found at {samples_path}")

        records: list[dict[str, Any]] = []
        with open(samples_path, encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON on line {line_idx + 1} of {samples_path}"
                    ) from exc

        if not records:
            raise ValueError(f"No records found in {samples_path}")

        return cls.from_records(
            records, run_dir=run_path if run_path.is_dir() else run_path.parent
        )

    @classmethod
    def from_records(
        cls,
        records: list[dict[str, Any]],
        run_dir: Path | None = None,
    ) -> ConditionOutcome:
        """Build ConditionOutcome from raw sample record dictionaries."""
        if not records:
            raise ValueError("Records list cannot be empty")

        parsed_items: list[
            tuple[int | str, bool, str, Any, str | None, str | None]
        ] = []
        conditions_found: set[str] = set()

        for idx, rec in enumerate(records):
            # Status check
            status = rec.get("status")
            if status != "complete":
                raise ValueError(
                    f"Incomplete sample encountered at index {idx} with status '{status}'"
                )

            # Sample ID extraction
            if "sample_index" in rec and rec["sample_index"] is not None:
                s_id: int | str = rec["sample_index"]
            elif "sample_id" in rec and rec["sample_id"] is not None:
                s_id = rec["sample_id"]
            else:
                raise ValueError(f"Record at index {idx} lacks sample_index/sample_id")

            # Condition extraction
            cond = rec.get("condition") or rec.get("condition_name")
            if not cond:
                identity = rec.get("identity")
                if isinstance(identity, dict):
                    cond = identity.get("condition")
            if not cond:
                raise ValueError(f"Record at index {idx} lacks condition identifier")
            conditions_found.add(str(cond))

            # Correctness
            corr = rec.get("correct")
            if corr is None:
                raise ValueError(f"Record at index {idx} lacks 'correct' field")
            corr_bool = bool(corr)

            # Dataset / split / gold extraction
            gold = rec.get("gold")
            identity = (
                rec.get("identity") if isinstance(rec.get("identity"), dict) else {}
            )
            if gold is None:
                gold = identity.get("gold")

            dataset = rec.get("dataset") or rec.get("task") or identity.get("task")
            split = rec.get("split") or identity.get("split")

            parsed_items.append((s_id, corr_bool, str(cond), gold, dataset, split))

        if len(conditions_found) > 1:
            raise ValueError(
                f"Multiple conditions found in records: {sorted(conditions_found)}"
            )
        condition = next(iter(conditions_found))

        # Check duplicate sample IDs
        all_ids = [item[0] for item in parsed_items]
        if len(all_ids) != len(set(all_ids)):
            seen = set()
            duplicates = set()
            for x in all_ids:
                if x in seen:
                    duplicates.add(x)
                seen.add(x)
            raise ValueError(
                f"Duplicate sample IDs found: {sorted(duplicates, key=str)}"
            )

        # Sort items by sample ID
        def _sort_key(
            item: tuple[int | str, bool, str, Any, str | None, str | None],
        ) -> tuple[int, int | str]:
            sid = item[0]
            if isinstance(sid, int):
                return (0, sid)
            if isinstance(sid, str) and sid.isdigit():
                return (0, int(sid))
            return (1, str(sid))

        parsed_items.sort(key=_sort_key)

        sorted_ids = [item[0] for item in parsed_items]
        correct_vector = [item[1] for item in parsed_items]
        golds = [item[3] for item in parsed_items]
        dataset_val = parsed_items[0][4]
        split_val = parsed_items[0][5]

        # Verify no missing IDs if numeric integer sequence
        int_ids = []
        for x in sorted_ids:
            if isinstance(x, int):
                int_ids.append(x)
            elif isinstance(x, str) and x.isdigit():
                int_ids.append(int(x))
            else:
                int_ids = None
                break

        if int_ids is not None and len(int_ids) > 0:
            min_id = min(int_ids)
            max_id = max(int_ids)
            expected_count = max_id - min_id + 1
            if len(int_ids) != expected_count:
                missing = set(range(min_id, max_id + 1)) - set(int_ids)
                raise ValueError(
                    f"Missing sample IDs detected in numeric sequence: {sorted(missing)[:10]}"
                )

        outcome = cls(
            condition=condition,
            sample_ids=sorted_ids,
            correct=correct_vector,
            dataset=dataset_val,
            split=split_val,
            golds=golds,
        )

        # Check summary.json if available
        if run_dir is not None:
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                cls._validate_against_summary(outcome, summary_path)

        return outcome

    @staticmethod
    def _validate_against_summary(
        outcome: ConditionOutcome, summary_path: Path
    ) -> None:
        """Verify that outcome accuracy and counts match summary.json."""
        with open(summary_path, encoding="utf-8") as f:
            summary_data = json.load(f)

        cond = outcome.condition
        stats = None
        if (
            "condition_statistics" in summary_data
            and cond in summary_data["condition_statistics"]
        ):
            stats = summary_data["condition_statistics"][cond]
        elif "metrics" in summary_data:
            m = summary_data["metrics"]
            acc_key = f"{cond}_accuracy"
            corr_key = f"{cond}_correct"
            if acc_key in m:
                stats = {"accuracy": m[acc_key], "correct": m.get(corr_key)}

        if stats is not None:
            expected_acc = stats.get("accuracy")
            if (
                expected_acc is not None
                and abs(outcome.accuracy - float(expected_acc)) > 1e-6
            ):
                raise ValueError(
                    f"Accuracy mismatch with summary.json: outcome={outcome.accuracy}, summary={expected_acc}"
                )
            expected_corr = stats.get("correct")
            if expected_corr is not None and outcome.n_correct != int(expected_corr):
                raise ValueError(
                    f"Correct count mismatch with summary.json: outcome={outcome.n_correct}, summary={expected_corr}"
                )
