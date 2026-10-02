"""Read communication inputs exclusively from explicit MLflow run artifacts."""

import json
import tempfile
from pathlib import Path

import numpy as np

from .stepwise_communication_analysis import MODELS, grid, number


class SavedRunReader:
    def __init__(self, client, directory):
        self.client = client
        self.directory = directory
        self.provenance = []

    def run(self, run_id, experiment):
        run = self.client.get_run(run_id)
        name = self.client.get_experiment(run.info.experiment_id).name
        if run.info.status != "FINISHED" or name != experiment:
            raise ValueError(f"Expected FINISHED {experiment} run: {run_id}")
        return run

    def artifact(self, run_id, path):
        destination = self.directory / run_id
        destination.mkdir(exist_ok=True)
        downloaded = Path(
            self.client.download_artifacts(run_id, path, str(destination))
        )
        self.provenance.append({"run_id": run_id, "artifact_path": path})
        if path.endswith(".jsonl"):
            return [
                json.loads(line) for line in downloaded.read_text().splitlines() if line
            ]
        return json.loads(downloaded.read_text())

    def probes(self, run_id, sweep_ids):
        run = self.run(run_id, "latentmas_receiver_trajectory")
        source_id = run.data.params.get("source_run_id") or run.data.tags.get(
            "source_run_id"
        )
        if source_id not in sweep_ids:
            raise ValueError(
                f"Probe source is not a selected receiver sweep: {run_id}, {source_id}"
            )
        if (
            run.data.tags.get("phase") != "sender_probe"
            or run.data.params.get("source_artifact") != "sender_latent_states.pt"
        ):
            raise ValueError(
                f"Expected a probe of the sweep's saved sender states: {run_id}"
            )
        source = self.run(source_id, "latentmas_receiver_trajectory")
        if source.data.tags.get("phase") != "sweep":
            raise ValueError(f"Probe source must be a trajectory sweep: {source_id}")
        metadata = {**source.data.tags, **source.data.params}
        model = metadata.get("model")
        steps = metadata.get("latent_steps", metadata.get("sender_latent_steps"))
        if model not in MODELS or str(steps) != "20":
            raise ValueError(f"Expected supported full20 probe source: {source_id}")
        results = self.artifact(run_id, "probe/results.json")
        stats = self.artifact(run_id, "probe/null_statistics.json")
        order = stats["cell_order"]
        null = np.asarray(stats["permutation_accuracies"], dtype=float)
        if (
            null.ndim != 2
            or null.shape[0] == 0
            or null.shape[1] != len(order)
            or not np.isfinite(null).all()
            or not ((null >= 0) & (null <= 1)).all()
        ):
            raise ValueError(f"Invalid saved probe null statistics: {run_id}")
        observed = {}
        for cell in results["cells"]:
            key = (cell["representation"], int(cell["step"]))
            if key in observed:
                raise ValueError(f"Duplicate saved probe cell: {run_id}, {key}")
            observed[key] = cell
        keys = [(cell["representation"], int(cell["step"])) for cell in order]
        if len(set(keys)) != len(keys) or set(keys) != set(observed):
            raise ValueError(f"Probe cell/null order mismatch: {run_id}")
        rows = []
        for index, key in enumerate(keys):
            if key[0] == "latent_post_realign":
                cell = observed[key]
                rows.append(
                    {
                        "model": model,
                        "latent_steps": "20",
                        "step": str(key[1]),
                        "representation": key[0],
                        "probe_accuracy": number(cell, "observed_oof_accuracy"),
                        "null_mean_accuracy": float(null[:, index].mean()),
                        "fwer_p_value": number(cell, "fwer_p_value"),
                        "probe_source_run_id": source_id,
                    }
                )
        self.provenance.append(
            {"probe_run_id": run_id, "source_run_id": source_id, "model": model}
        )
        return rows

    def receiver(self, run_id, sweep):
        experiment = (
            "latentmas_receiver_trajectory"
            if sweep
            else "latentmas_receiver_acquisition"
        )
        run = self.run(run_id, experiment)
        metadata = {**run.data.tags, **run.data.params}
        model = metadata.get("model")
        if model not in MODELS:
            raise ValueError(f"Unsupported receiver model: {run_id}")
        path = "sample_results.jsonl" if sweep else "results/sample_results.jsonl"
        records = self.artifact(run_id, path)
        if sweep:
            source = self.artifact(run_id, "source.json")
            parity = self.artifact(run_id, "parity.json")
            if (
                source["smoke"] is not False
                or source["sample_count"] != 100
                or parity["passed"] is not True
            ):
                raise ValueError(f"Incomplete or unverified sweep: {run_id}")
            steps = range(1, 21)
        else:
            step = int(metadata["latent_steps"])
            if step not in (1, 4, 20):
                raise ValueError(f"Expected independent step 1/4/20: {run_id}")
            steps = (step,)
        drops, own = {}, {}
        for record in records:
            if (
                record["error"] is not None
                or record["model"] != model
                or record["seed"] != 42
            ):
                raise ValueError(f"Invalid receiver observation: {run_id}")
            condition = record["condition"]
            selected = condition == "drop" or (
                condition == "own" and record["context_mode"] == "latent_only"
            )
            if not selected:
                if sweep:
                    raise ValueError(f"Unexpected sweep condition: {run_id}")
                continue
            digit = record["predicted_digit"]
            if type(digit) is not int or digit not in range(10):
                raise ValueError(f"Invalid receiver prediction: {run_id}")
            sample_id = record["sample_id"]
            key = (
                sample_id
                if condition == "drop"
                else (int(record["latent_steps"]), sample_id)
            )
            target = drops if condition == "drop" else own
            if key in target:
                raise ValueError(f"Duplicate receiver observation: {run_id}, {key}")
            target[key] = record
        if len(drops) != 100 or len(own) != 100 * len(steps):
            raise ValueError(f"Incomplete receiver sample set: {run_id}")
        from latentmas_reprop.application.receiver_acquisition.sampling import (
            generate_secret_digit_samples,
        )

        canonical = {
            sample.sample_id: sample
            for sample in generate_secret_digit_samples(100, 42)
        }
        for record in (*drops.values(), *own.values()):
            sample = canonical.get(record["sample_id"])
            if sample is None or (record["sample_key"], record["target_digit"]) != (
                sample.sample_key,
                sample.digit,
            ):
                raise ValueError(f"Noncanonical receiver identity: {run_id}")
        rows = []
        for step in steps:
            cells = {sample_id: row for (k, sample_id), row in own.items() if k == step}
            if set(cells) != set(drops):
                raise ValueError(f"Unpaired receiver step: {run_id}, {step}")
            for row in cells.values():
                if (
                    row["source_digit"] != row["target_digit"]
                    or row["cache_sequence_length"] != step
                    or row["receiver_position_start"] != row["original_full_seq_len"]
                    or row["retained_tail_start_position"] + step
                    != row["original_full_seq_len"]
                    or row["retained_original_position_end"]
                    != row["original_full_seq_len"]
                ):
                    raise ValueError(f"Invalid latent-only handoff: {run_id}, {step}")
            changed = (
                sum(
                    row["predicted_digit"] != drops[sample_id]["predicted_digit"]
                    for sample_id, row in cells.items()
                )
                / 100
            )
            rows.append(
                {
                    "model": model,
                    "latent_step": step,
                    "latent_steps": step,
                    "run_id": run_id,
                    "condition": "latent_only/own",
                    "receiver_changed_fraction": changed,
                    "argmax_changed_fraction": changed,
                }
            )
        return rows


def load_inputs(client, probe_ids, sweep_ids, independent_ids):
    """Download each selected artifact afresh into temporary storage, with no fallback."""
    for ids, count in ((probe_ids, 3), (sweep_ids, 3), (independent_ids, 9)):
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"Expected {count} unique explicit MLflow run IDs")
    with tempfile.TemporaryDirectory(prefix="communication-inputs-") as temporary:
        reader = SavedRunReader(client, Path(temporary))
        probes = [
            row for run_id in probe_ids for row in reader.probes(run_id, set(sweep_ids))
        ]
        probe_sources = [
            entry["source_run_id"]
            for entry in reader.provenance
            if "probe_run_id" in entry
        ]
        if len(set(probe_sources)) != len(sweep_ids):
            raise ValueError("Expected exactly one sender probe per selected sweep")
        sweep = [row for run_id in sweep_ids for row in reader.receiver(run_id, True)]
        independent = [
            row for run_id in independent_ids for row in reader.receiver(run_id, False)
        ]
        grid(sweep, "latent_step", range(1, 21), "receiver sweep")
        grid(independent, "latent_steps", (1, 4, 20), "independent receiver")
        return probes, sweep, independent, reader.provenance
