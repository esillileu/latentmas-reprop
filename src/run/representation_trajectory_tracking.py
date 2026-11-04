"""Track representation trajectories and their evidence."""

import traceback
from contextlib import contextmanager
from pathlib import Path

from latentmas_reprop.domain.models.use_cases import (
    REPRESENTATION_TRAJECTORY,
    mlflow_experiment_name,
)
from latentmas_reprop.domain.services.prompts.acquisition import (
    CANDIDATE_DIGITS,
    RECEIVER_ANSWER_PREFIX,
    RECEIVER_PROMPT,
    RECEIVER_PROMPT_TEMPLATE_VERSION,
    SENDER_PROMPT_TEMPLATE,
    SENDER_PROMPT_TEMPLATE_VERSION,
)
from latentmas_reprop.infrastructure.tracking import MLflowTracker


@contextmanager
def tracked_model(directory, name, smoke, exp_rel=None):
    """Keep parity evidence and trajectories in a separate MLflow experiment."""
    rel_name = exp_rel or Path(directory).name
    tracker = MLflowTracker()
    tracker.start_run(
        experiment_name=mlflow_experiment_name(REPRESENTATION_TRAJECTORY),
        run_name=rel_name,
        tags={
            "protocol": "representation_acquisition",
            "phase": "smoke" if smoke else "sweep",
            "experiment_type": REPRESENTATION_TRAJECTORY,
            "model": name,
            "task": "secret_digit",
            "method": "latent_mas",
            "seed": "42",
            "latent_steps": "20",
            "data_dir": str(Path(directory).resolve()),
            "rel_dir": rel_name,
        },
    )
    tracker.log_params(
        {
            "model": name,
            "sender_latent_steps": 20,
            "handoff_cuts": list(range(1, 21)),
            "sample_count": 10 if smoke else 100,
            "seed": 42,
            "dtype": "bfloat16",
            "condition": "latent_only/own + matched drop",
            "smoke": smoke,
            "task": "secret_digit",
            "method": "latent_mas",
            "backend": "transformers",
            "device": "cuda",
            "config_path": None,
            "context_modes": "latent_only",
            "carrier_modes": "latent_only",
            "acquisition_conditions": "own,drop",
            "candidate_digits": list(CANDIDATE_DIGITS),
            "sender_prompt_template_version": SENDER_PROMPT_TEMPLATE_VERSION,
            "sender_prompt_template": SENDER_PROMPT_TEMPLATE,
            "receiver_prompt_template_version": RECEIVER_PROMPT_TEMPLATE_VERSION,
            "receiver_prompt_template": RECEIVER_PROMPT,
            "receiver_answer_prefix": RECEIVER_ANSWER_PREFIX,
            "save_hidden_states": False,
            "save_raw_cache": False,
            "latent_space_realign": False,
            "probe_sender_latents": True,
            "probe_prompt_templates": 20,
            "save_latent_states": True,
            "save_receiver_logits": True,
        }
    )
    status = "FAILED"
    try:
        yield tracker
        status = "FINISHED"
    except (Exception, KeyboardInterrupt):
        tracker.log_params({"failure_traceback": traceback.format_exc()[:500]})
        tracker.log_dict({"traceback": traceback.format_exc()}, "failure.json")
        raise
    finally:
        for filename in (
            "parity.json",
            "sample_results.jsonl",
            "sample_results.json",
            "sender_contexts.jsonl",
            "sender_probe_inputs.jsonl",
            "sender_latent_states.pt",
            "candidate_token_mapping.json",
            "source.json",
            "resolved_config.json",
            "receiver_input.json",
        ):
            path = directory / filename
            if path.is_file():
                tracker.log_artifact(path)
        for path in sorted((directory / "receiver_outputs").glob("*.pt")):
            tracker.log_artifact(path, artifact_path="receiver_outputs")
        tracker.flush_traces()
        tracker.end_run(status)
