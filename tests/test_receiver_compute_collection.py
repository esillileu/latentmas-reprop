"""Receiver collection reuses paired contexts and baseline trajectories."""

from types import SimpleNamespace

from src.run.cli import parse_run_matrix


def test_collection_builds_each_context_once_and_uses_other_sample(monkeypatch):
    import latentmas_reprop.application.receiver_compute_preflight.trajectory as trajectory
    import latentmas_reprop.application.receiver_compute_preflight_use_case as module

    class Tracker:
        def __init__(self):
            self.artifacts = {}
            self.status = None

        def start_run(self, **kwargs):
            pass

        def log_params(self, config):
            self.config = config

        def log_artifact(self, path, **kwargs):
            self.artifacts[path.name] = path.read_text()

        def log_metrics(self, metrics):
            pass

        def flush_traces(self):
            pass

        def end_run(self, status):
            self.status = status

    class Dataset:
        def load(self, **kwargs):
            return [
                {"question": "first", "gold": "1"},
                {"question": "second", "gold": "2"},
            ]

    tracker = Tracker()
    args = parse_run_matrix(
        [
            "-c",
            "lmas/receiver_compute_preflight/gsm8k",
            "--max_samples",
            "2",
            "--receiver_budgets",
            "2,4,free",
            "--max_new_tokens",
            "8",
            "--bootstrap_count",
            "10",
        ]
    )[0]
    calls, generation_calls = [], []

    def build(method, item, step, width, context_id, tracker, runtime):
        assert width is None
        calls.append((item["question"], step))
        return (
            None,
            [[{"latent_steps": step}]],
            {"handoff_positions": len(item["question"])},
        )

    def generate(method, item, context, limit, budgets=(), report_progress=False):
        generation_calls.append((item["question"], method.latent_steps, limit))
        return [1, 2][:limit], {
            "receiver_input_ids": [5],
            "receiver_prompt": "prompt",
            "receiver_thinking_open": False,
            "receiver_cache_positions": 0,
            "receiver_prompt_tokens": 1,
            "receiver_latency_sec": 0.1,
            "receiver_budget_latency_sec": {str(b): 0.05 for b in budgets},
        }

    monkeypatch.setattr(module, "build_upstream", build)
    monkeypatch.setattr(trajectory, "generate_receiver", generate)
    model = SimpleNamespace(
        device="cpu",
        tokenizer=SimpleNamespace(decode=lambda ids, **kwargs: str(ids[-1])),
    )
    metrics, rows = module.ReceiverComputePreflightUseCase(
        dataset_port=Dataset(), tracker_port=tracker
    ).execute(model, args)
    assert len(calls) == 4
    assert len(generation_calls) == 10
    assert sum(u == 0 for _, u, _ in generation_calls) == 2
    assert len(rows) == 36
    assert metrics["sample_count"] == 2
    assert tracker.status == "FINISHED"
    assert "sample_results.jsonl" in tracker.artifacts
    assert all(
        r["sample_id"] != r["donor_id"]
        for r in rows
        if r["handoff_condition"] == "mismatched"
    )
    assert all(r["valid"] for r in rows)

    mismatched = [r for r in rows if r["handoff_condition"] == "mismatched"]
    assert all(r["donor_length_abs_delta"] == 1 for r in mismatched)
    assert {r["donor_length_delta"] for r in mismatched} == {-1, 1}
    assert all(
        r["mean_receiver_latency_sec"] == 0.05
        for r in metrics["curves"]
        if r["receiver_budget"] == 2
    )
