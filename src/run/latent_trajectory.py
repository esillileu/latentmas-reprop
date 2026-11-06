"""Collect training-free latent computation trajectories."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv

from latentmas_reprop.application.latent_trajectory import collect
from latentmas_reprop.infrastructure.datasets.registry import DEFAULT_DATASET_REGISTRY
from latentmas_reprop.infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
from latentmas_reprop.infrastructure.storage import save_json_atomic

from .latent_trajectory_tracking import sample_run_name, tracked_sample


def main(argv=None):
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default="latent_trajectory/gsm8k_q38")
    preliminary, _ = parser.parse_known_args(argv)
    path = Path("configs") / f"{preliminary.config}.yaml"
    defaults = yaml.safe_load(path.read_text())
    for name in (
        "model_name",
        "task",
        "split",
        "prompt",
        "initial_input",
        "initial_role",
    ):
        parser.add_argument(f"--{name}", default=defaults[name])
    for name in (
        "max_samples",
        "latent_steps",
        "latent_step_start",
        "max_new_tokens",
        "seed",
    ):
        parser.add_argument(f"--{name}", type=int, default=defaults[name])
    for name in ("temperature", "top_p"):
        parser.add_argument(f"--{name}", type=float, default=defaults[name])
    parser.add_argument(
        "--conditions", nargs="+", choices=list("abcd"), default=defaults["conditions"]
    )
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument(
        "--latent_space_realign",
        action="store_true",
        help="Enable latent space realignment.",
    )
    alignment.add_argument(
        "--no_latent_space_realign",
        dest="latent_space_realign",
        action="store_false",
        help="Disable latent space realignment, overriding the preset.",
    )
    parser.set_defaults(latent_space_realign=defaults["latent_space_realign"])
    parser.add_argument("--model_revision")
    parser.add_argument(
        "--sample_indices",
        nargs="+",
        type=int,
        default=defaults.get("sample_indices"),
        help="Zero-based dataset indices; overrides max_samples and runs in dataset order.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("storage/experiments/latent_trajectory"),
    )
    parser.add_argument("--no_progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_indices is not None:
        if not args.sample_indices or any(index < 0 for index in args.sample_indices):
            parser.error("sample_indices must contain non-negative dataset indices")
        args.sample_indices = sorted(set(args.sample_indices))
    if not 0 <= args.latent_step_start <= args.latent_steps:
        parser.error("Require 0 <= latent_step_start <= latent_steps")
    if args.max_new_tokens <= 0 or args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("Invalid decoding settings")
    if args.initial_input != "rq":
        parser.error("This experiment requires initial_input=rq")
    alignments = args.latent_space_realign
    alignments = alignments if isinstance(alignments, list) else [alignments]
    if not alignments or any(type(value) is not bool for value in alignments):
        parser.error(
            "latent_space_realign must be a boolean or non-empty list of booleans"
        )
    runs = []
    for enabled in dict.fromkeys(alignments):
        run_args = argparse.Namespace(**vars(args))
        run_args.latent_space_realign = enabled
        runs.append(run_args)
    plans = [run_plan(run_args) for run_args in runs]
    if args.dry_run:
        print(json.dumps(plans[0] if len(plans) == 1 else {"runs": plans}, indent=2))
        return
    for run_args, plan in zip(runs, plans, strict=True):
        run_collection(run_args, plan["config"], Path(plan["directory"]))


def run_plan(args):
    """Resolve one scalar configuration and its persistent directory."""
    config = {
        k: v
        for k, v in vars(args).items()
        if k
        not in (
            "config",
            "output_root",
            "dry_run",
            "no_progress",
            "max_samples",
            "conditions",
            "sample_indices",
        )
    }
    # Step zero extends the original readout range without changing its cache/run.
    identity_config = {**config, "latent_step_start": max(1, args.latent_step_start)}
    identity = hashlib.sha256(
        json.dumps(identity_config, sort_keys=True).encode()
    ).hexdigest()[:16]
    directory = args.output_root / args.task / args.model_name.split("/")[-1] / identity
    indices = args.sample_indices
    if indices is None and args.max_samples > 0:
        indices = range(args.max_samples)
    return {
        "config": config,
        "conditions": args.conditions,
        "max_samples": args.max_samples,
        "sample_indices": args.sample_indices,
        "directory": str(directory),
        "mlflow_runs": [
            {"sample_index": index, "run_name": sample_run_name(directory, args, index)}
            for index in indices
        ]
        if indices is not None
        else None,
    }


def run_collection(args, config, directory):
    """Execute one independent model/alignment run."""
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelWrapper(args.model_name, device, args=args)
    model.model.eval()
    from latentmas_reprop.application.intervention_continuation.decoder import (
        generation_settings,
    )
    from latentmas_reprop.application.intervention_continuation.identity import (
        ensure_state,
        state_identity,
    )

    identity, provenance = state_identity(model, args, "transformers")
    ensure_state(directory, identity, provenance)

    save_json_atomic(
        directory / "metadata.json",
        {
            **config,
            "effective_decoding": generation_settings(model, args),
            "model_revision": getattr(model.model.config, "_commit_hash", None),
            "tokenizer_revision": model.tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer": model.tokenizer.name_or_path,
            "dtype": str(model.dtype),
            "chat_template": model.tokenizer.chat_template,
            "torch_version": torch.__version__,
            "state_identity": identity,
            "provenance": provenance,
        },
    )
    count = collect(
        model,
        args,
        DEFAULT_DATASET_REGISTRY,
        DEFAULT_EVALUATOR,
        directory,
        sample_context=partial(
            tracked_sample, directory=directory, args=args, config=config
        ),
    )
    print(f"Collected {count} samples in {directory}")


if __name__ == "__main__":
    main()
