"""Evaluate full20 latent-only KV prefixes; BF16, parity-gated, no vector injection."""

import argparse
import gc
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import transformers
from dotenv import load_dotenv

from latentmas_reprop.application.receiver_acquisition.parity import (
    verify_prefix_parity,
)
from latentmas_reprop.application.receiver_acquisition.sampling import (
    generate_secret_digit_samples,
)
from latentmas_reprop.application.receiver_acquisition.scoring import (
    prepare_receiver_scoring,
)
from latentmas_reprop.application.receiver_acquisition.trajectory import (
    collect_trajectory,
)
from latentmas_reprop.application.sender_probe import collect_sender_states
from latentmas_reprop.domain.services.prompts.acquisition import (
    RECEIVER_ANSWER_PREFIX,
    RECEIVER_PROMPT,
    RECEIVER_PROMPT_TEMPLATE_VERSION,
    SENDER_PROMPT_TEMPLATE,
    SENDER_PROMPT_TEMPLATE_VERSION,
)
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
from latentmas_reprop.infrastructure.paths.resolver import get_git_commit_hash
from latentmas_reprop.infrastructure.tracking import MLflowTracker


@contextmanager
def tracked_model(directory, name, smoke):
    """Keep parity evidence and trajectories in a separate MLflow experiment."""
    tracker = MLflowTracker()
    tracker.start_run(
        experiment_name="latentmas_receiver_trajectory",
        run_name=f"{name.split('/')[-1]}-{'parity' if smoke else 'steps-1-20'}",
        tags={
            "protocol": "receiver_acquisition",
            "phase": "parity" if smoke else "sweep",
        },
    )
    tracker.log_params(
        {
            "model": name,
            "sender_latent_steps": 20,
            "handoff_cuts": list(range(1, 21)),
            "sample_count": 100,
            "seed": 42,
            "dtype": "bfloat16",
            "condition": "latent_only/own + matched drop",
            "parity_only": smoke,
        }
    )
    status = "FAILED"
    try:
        yield tracker
        status = "FINISHED"
    finally:
        for filename in (
            "parity.json",
            "sample_results.jsonl",
            "sender_latent_states.pt",
            "candidate_token_mapping.json",
            "source.json",
            "resolved_config.json",
            "receiver_input.json",
        ):
            path = directory / filename
            if path.is_file():
                tracker.log_artifact(path)
        tracker.flush_traces()
        tracker.end_run(status)


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="0.6B parity only; ten balanced representatives from the canonical 100 samples.",
    )
    parser.add_argument("--model", choices=("0.6B", "4B", "8B", "14B", "all"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    git_commit = get_git_commit_hash()
    if args.smoke and args.model not in (None, "0.6B"):
        parser.error("Local smoke is restricted to 0.6B")
    device = torch.device("cuda")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires a BF16-capable CUDA device")
    if (
        not args.smoke
        and args.model != "0.6B"
        and torch.cuda.get_device_properties(device).total_memory < 30 * 1024**3
    ):
        raise RuntimeError(
            "4B/8B/14B trajectories require a 32GB GPU; use --model 0.6B locally"
        )
    models = (
        ["0.6B"]
        if args.smoke
        else (
            [args.model]
            if args.model not in (None, "all")
            else ["0.6B", "4B", "8B", "14B"]
        )
    )
    samples = generate_secret_digit_samples(100, 42)
    representatives = [
        next(s for s in samples if s.digit == digit) for digit in range(10)
    ]
    output = args.output_dir or Path(
        "artifacts/receiver_trajectory" + ("/smoke" if args.smoke else "")
    )
    torch.manual_seed(42)
    for size in models:
        name = f"Qwen/Qwen3-{size}"
        directory = output / name.replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        with tracked_model(directory, name, args.smoke) as tracker:
            model = ModelWrapper(name, device, model_dtype=torch.bfloat16)
            receiver = prepare_receiver_scoring(model)
            (directory / "resolved_config.json").write_text(
                json.dumps(
                    {
                        k: str(v) if isinstance(v, Path) else v
                        for k, v in vars(args).items()
                    },
                    indent=2,
                )
                + "\n"
            )
            (directory / "receiver_input.json").write_text(
                json.dumps(
                    {
                        "input_ids": receiver[0].cpu().tolist(),
                        "attention_mask": receiver[1].cpu().tolist(),
                    },
                    indent=2,
                )
                + "\n"
            )
            parity = verify_prefix_parity(model, representatives, receiver)
            parity.update(
                model=name,
                dtype="bfloat16",
                gpu=torch.cuda.get_device_name(device),
                git_commit=git_commit,
                torch_version=torch.__version__,
                transformers_version=transformers.__version__,
            )
            (directory / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
            if not parity["passed"]:
                raise ValueError(
                    f"Exact parity failed; no trajectory was run. See {directory / 'parity.json'}"
                )
            tracker.log_metrics({"parity_passed": 1})
            if args.smoke:
                del model, receiver
                gc.collect()
                torch.cuda.empty_cache()
                continue
            selected = samples
            records = collect_trajectory(
                model, selected, receiver, directory / "sample_results.jsonl", tracker
            )
            states = collect_sender_states(model, latent_steps=20, template_count=20)
            torch.save(states.payload(), directory / "sender_latent_states.pt")
            (directory / "candidate_token_mapping.json").write_text(
                json.dumps(receiver[2], indent=2) + "\n"
            )
            (directory / "source.json").write_text(
                json.dumps(
                    {
                        "model": name,
                        "sample_count": len(selected),
                        "seed": 42,
                        "dtype": "bfloat16",
                        "handoff": "full20[:prompt_len+k] then last k positions",
                        "receiver_position_start": "prompt_len+k",
                        "sender_prompt_template": SENDER_PROMPT_TEMPLATE,
                        "sender_prompt_template_version": SENDER_PROMPT_TEMPLATE_VERSION,
                        "receiver_prompt_template": RECEIVER_PROMPT,
                        "receiver_prompt_template_version": RECEIVER_PROMPT_TEMPLATE_VERSION,
                        "receiver_answer_prefix": RECEIVER_ANSWER_PREFIX,
                        "backend": "transformers",
                        "device": str(device),
                        "transformers_version": transformers.__version__,
                        "torch_version": torch.__version__,
                        "sender_state_templates": 20,
                        "sender_state_examples": 200,
                        "save_hidden_states": False,
                        "save_raw_cache": False,
                        "latent_space_realign": False,
                        "smoke": args.smoke,
                        "gpu": torch.cuda.get_device_name(device),
                        "git_commit": git_commit,
                    },
                    indent=2,
                )
                + "\n"
            )
            tracker.log_metrics(
                {
                    "count/observations": len(records),
                    "count/sender_state_examples": len(states.labels),
                }
            )
            del states, records
        del model, receiver
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Saved receiver trajectory: {output}")


if __name__ == "__main__":
    main()
