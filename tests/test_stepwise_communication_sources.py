"""MLflow is the only input authority; verify saved observation validation."""

import copy
import json
from types import SimpleNamespace

import pytest

from latentmas_reprop.application.receiver_acquisition.sampling import (
    generate_secret_digit_samples,
)
from src.run import stepwise_communication_analysis as analysis
from src.run import stepwise_communication_sources as sources


class SavedClient:
    def __init__(self):
        self.runs, self.payloads, self.downloads = {}, {}, []
        self.names = {
            "probe": "latentmas_receiver_trajectory",
            "sweep": "latentmas_receiver_trajectory",
            "independent": "latentmas_receiver_acquisition",
        }
        self.probe_ids, self.sweep_ids, self.independent_ids = [], [], []
        for index, model in enumerate(analysis.MODELS[:3]):
            probe_id, sweep_id = f"probe-{index}", f"sweep-{index}"
            self.probe_ids.append(probe_id)
            self.sweep_ids.append(sweep_id)
            self.add_run(
                sweep_id, "sweep", {"model": model, "sender_latent_steps": "20"}
            )
            self.add_run(
                probe_id,
                "probe",
                {
                    "source_run_id": sweep_id,
                    "source_artifact": "sender_latent_states.pt",
                },
            )
            cells = [
                {
                    "representation": "latent_post_realign",
                    "step": step,
                    "observed_oof_accuracy": 0.2,
                    "fwer_p_value": 0.049,
                }
                for step in range(1, 21)
            ]
            self.payloads[probe_id, "probe/results.json"] = {"cells": cells}
            self.payloads[probe_id, "probe/null_statistics.json"] = {
                "cell_order": [
                    {"representation": cell["representation"], "step": cell["step"]}
                    for cell in cells
                ],
                "permutation_accuracies": [[0.1] * 20, [0.12] * 20],
            }
            self.payloads[sweep_id, "source.json"] = {
                "smoke": False,
                "sample_count": 100,
            }
            self.payloads[sweep_id, "parity.json"] = {"passed": True}
            self.payloads[sweep_id, "sample_results.jsonl"] = self.records(
                model, range(1, 21)
            )
            for step in (1, 4, 20):
                run_id = f"independent-{index}-{step}"
                self.independent_ids.append(run_id)
                self.add_run(
                    run_id, "independent", {"model": model, "latent_steps": str(step)}
                )
                self.payloads[run_id, "results/sample_results.jsonl"] = self.records(
                    model, (step,)
                )

    def add_run(self, run_id, experiment, params):
        self.runs[run_id] = SimpleNamespace(
            info=SimpleNamespace(status="FINISHED", experiment_id=experiment),
            data=SimpleNamespace(
                params=params,
                tags={"phase": "sender_probe" if experiment == "probe" else "sweep"},
            ),
        )

    @staticmethod
    def records(model, steps):
        rows = []
        for sample in generate_secret_digit_samples(100, 42):
            base = {
                "sample_id": sample.sample_id,
                "sample_key": sample.sample_key,
                "target_digit": sample.digit,
                "source_digit": sample.digit,
                "model": model,
                "seed": 42,
                "error": None,
                "predicted_digit": 5,
            }
            rows.append(base | {"condition": "drop"})
            for step in steps:
                rows.append(
                    base
                    | {
                        "condition": "own",
                        "latent_steps": step,
                        "context_mode": "latent_only",
                        "cache_sequence_length": step,
                        "receiver_position_start": 34 + step,
                        "original_full_seq_len": 34 + step,
                        "retained_tail_start_position": 34,
                        "retained_original_position_end": 34 + step,
                        "predicted_digit": 4 if sample.digit < 3 else 5,
                    }
                )
        return rows

    def get_run(self, run_id):
        return self.runs[run_id]

    def get_experiment(self, experiment_id):
        return SimpleNamespace(name=self.names[experiment_id])

    def download_artifacts(self, run_id, path, destination):
        from pathlib import Path

        self.downloads.append((run_id, path, destination))
        target = Path(destination) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads[run_id, path]
        target.write_text(
            "\n".join(json.dumps(row) for row in payload)
            if path.endswith(".jsonl")
            else json.dumps(payload)
        )
        return str(target)


def test_mlflow_loads_all_inputs_and_removes_temporary_downloads():
    from pathlib import Path

    client = SavedClient()
    probes, sweep, independent, provenance = sources.load_inputs(
        client, client.probe_ids, client.sweep_ids, client.independent_ids
    )
    rows = analysis.communication_metrics(probes, sweep)
    parity = analysis.receiver_parity(rows, independent)
    assert len(rows) == 60 and len(parity) == 9
    assert all(row["matches"] for row in parity)
    assert all(row["receiver_changed_fraction"] == 0.3 for row in rows)
    assert provenance and client.downloads
    assert all(not Path(destination).exists() for _, _, destination in client.downloads)


@pytest.mark.parametrize(
    "corruption",
    (
        "duplicate",
        "missing",
        "nan",
        "identity",
        "position",
        "smoke",
        "status",
        "probe_duplicate",
        "null_nan",
        "null_duplicate",
    ),
)
def test_mlflow_corrupt_inputs_raise_instead_of_skipping(corruption):
    client = SavedClient()
    rows = client.payloads["sweep-0", "sample_results.jsonl"]
    if corruption == "duplicate":
        rows[22] = copy.deepcopy(rows[1])
    elif corruption == "missing":
        rows.pop()
    elif corruption == "nan":
        rows[1]["predicted_digit"] = float("nan")
    elif corruption == "identity":
        rows[1]["sample_key"] = "wrong"
    elif corruption == "position":
        rows[1]["receiver_position_start"] += 20
    elif corruption == "smoke":
        client.payloads["sweep-0", "source.json"]["smoke"] = True
    elif corruption == "status":
        client.runs["sweep-0"].info.status = "FAILED"
    elif corruption == "probe_duplicate":
        cells = client.payloads["probe-0", "probe/results.json"]["cells"]
        cells[-1] = copy.deepcopy(cells[0])
    elif corruption == "null_nan":
        client.payloads["probe-0", "probe/null_statistics.json"][
            "permutation_accuracies"
        ][0][0] = float("nan")
    else:
        order = client.payloads["probe-0", "probe/null_statistics.json"]["cell_order"]
        order[-1] = copy.deepcopy(order[0])
    with pytest.raises(ValueError):
        sources.load_inputs(
            client, client.probe_ids, client.sweep_ids, client.independent_ids
        )


def test_missing_artifact_has_no_local_fallback():
    client = SavedClient()
    del client.payloads["sweep-0", "sample_results.jsonl"]
    with pytest.raises(KeyError):
        sources.load_inputs(
            client, client.probe_ids, client.sweep_ids, client.independent_ids
        )


def test_cli_uploads_derived_artifacts_to_mlflow(tmp_path, monkeypatch):
    import mlflow.tracking

    client = SavedClient()
    uploads, terminated = [], []
    client.get_experiment_by_name = lambda name: SimpleNamespace(experiment_id="sweep")
    client.create_run = lambda *args, **kwargs: SimpleNamespace(
        info=SimpleNamespace(run_id="analysis")
    )
    client.log_artifact = lambda run_id, path, directory: uploads.append(
        (run_id, path, directory)
    )
    client.set_terminated = lambda run_id, status: terminated.append((run_id, status))
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(analysis, "plot_metrics", lambda *args: None)
    analysis.main(
        [
            "--tracking-uri",
            "https://tracking.example",
            "--probe-run-ids",
            *client.probe_ids,
            "--sweep-run-ids",
            *client.sweep_ids,
            "--independent-run-ids",
            *client.independent_ids,
        ]
    )
    assert len(uploads) == 3
    assert all(
        run_id == "analysis" and directory == "communication"
        for run_id, _, directory in uploads
    )
    assert terminated == [("analysis", "FINISHED")]


@pytest.mark.parametrize(
    "corruption", ("acquisition", "other_sweep", "wrong_artifact", "duplicate_source")
)
def test_probe_cannot_mix_sources(corruption):
    client = SavedClient()
    probe = client.runs["probe-0"]
    if corruption == "acquisition":
        client.names["probe"] = "latentmas_sender_probe"
        probe.data.params["source_run_id"] = "independent-0-20"
    elif corruption == "other_sweep":
        probe.data.params["source_run_id"] = "unselected-sweep"
    elif corruption == "wrong_artifact":
        probe.data.params["source_artifact"] = "probe/sender_latent_states.pt"
    else:
        probe.data.params["source_run_id"] = "sweep-1"
    with pytest.raises(ValueError):
        sources.load_inputs(
            client, client.probe_ids, client.sweep_ids, client.independent_ids
        )
