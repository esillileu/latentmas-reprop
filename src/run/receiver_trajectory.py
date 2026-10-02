"""Evaluate full20 latent-only KV prefixes; BF16, parity-gated, no vector injection."""

import argparse
import csv
import gc
import json
from pathlib import Path

import torch
import transformers

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
    summarize_trajectory,
)
from latentmas_reprop.infrastructure.models.model_wrapper import ModelWrapper
from latentmas_reprop.infrastructure.paths.resolver import get_git_commit_hash


def load_history(acquisition, reference_dir, model):
    with (acquisition / "receiver_conditions.csv").open(newline="") as stream:
        cells = list(csv.DictReader(stream))
    history, sources = {}, {}
    for step in (1, 4):
        matches = [
            c
            for c in cells
            if c["model"] == model
            and c["latent_steps"] == str(step)
            and c["condition"] == "drop"
            and c["sample_count"] == "100"
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one canonical historical run for {model}, k={step}"
            )
        run_id = matches[0]["run_id"]
        path = reference_dir / run_id / "sample_results.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        selected = [
            r
            for r in records
            if r["condition"] == "drop"
            or (r["condition"] == "own" and r["context_mode"] == "latent_only")
        ]
        if len(selected) != 200 or any(
            r["model"] != model
            or r["latent_steps"] != step
            or r["error"]
            or r["seed"] != 42
            for r in selected
        ):
            raise ValueError(f"Invalid historical receiver run: {run_id}")
        for r in selected:
            if r["cache_present"] and r["cache_dtype"] != "torch.bfloat16":
                raise ValueError("Historical handoff must use BF16")
        history[step] = {(r["sample_id"], r["condition"]): r for r in selected}
        if len(history[step]) != 200:
            raise ValueError("Duplicate historical sample/condition")
        sources[str(step)] = run_id
    return history, sources


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="0.6B only; ten balanced representatives from the canonical 100 samples.",
    )
    parser.add_argument("--model", choices=("0.6B", "4B", "8B", "all"))
    parser.add_argument(
        "--acquisition-dir", type=Path, default=Path("artifacts/receiver_acquisition")
    )
    parser.add_argument(
        "--reference-dir", type=Path, default=Path(".cache/receiver_acquisition/mlflow")
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.smoke and args.model not in (None, "0.6B"):
        parser.error("Local smoke is restricted to 0.6B")
    device = torch.device("cuda")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires a BF16-capable CUDA device")
    if (
        not args.smoke
        and torch.cuda.get_device_properties(device).total_memory < 30 * 1024**3
    ):
        raise RuntimeError("Full trajectories require a 32GB GPU; use --smoke locally")
    models = (
        ["0.6B"]
        if args.smoke
        else ([args.model] if args.model not in (None, "all") else ["0.6B", "4B", "8B"])
    )
    if models[0] != "0.6B":
        gate = Path("artifacts/receiver_trajectory/Qwen_Qwen3-0.6B/parity.json")
        evidence = json.loads(gate.read_text()) if gate.exists() else {}
        if not evidence.get("passed") or any(
            evidence.get(key) != value
            for key, value in {
                "gpu": torch.cuda.get_device_name(device),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "git_commit": get_git_commit_hash(),
            }.items()
        ):
            raise ValueError(
                "First run 0.6B parity on this execution environment using --model 0.6B"
            )
    samples = generate_secret_digit_samples(100, 42)
    representatives = [
        next(s for s in samples if s.digit == digit) for digit in range(10)
    ]
    output = args.output_dir or Path(
        "artifacts/receiver_trajectory" + ("/smoke" if args.smoke else "")
    )
    with (args.acquisition_dir / "sender_probe_cells.csv").open(newline="") as stream:
        probes = list(csv.DictReader(stream))
    # Validate all historical inputs before loading any weights.
    historical = {
        m: load_history(args.acquisition_dir, args.reference_dir, f"Qwen/Qwen3-{m}")
        for m in models
    }
    combined = []
    torch.manual_seed(42)
    for size in models:
        name = f"Qwen/Qwen3-{size}"
        directory = output / name.replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        model = ModelWrapper(name, device, model_dtype=torch.bfloat16)
        receiver = prepare_receiver_scoring(model)
        history, sources = historical[size]
        parity = verify_prefix_parity(model, representatives, receiver, history)
        parity.update(
            model=name,
            dtype="bfloat16",
            historical_runs=sources,
            gpu=torch.cuda.get_device_name(device),
            git_commit=get_git_commit_hash(),
            torch_version=torch.__version__,
            transformers_version=transformers.__version__,
        )
        (directory / "parity.json").write_text(json.dumps(parity, indent=2) + "\n")
        if not parity["passed"]:
            raise ValueError(
                f"Exact parity failed; no trajectory was run. See {directory / 'parity.json'}"
            )
        selected = representatives if args.smoke else samples
        records = collect_trajectory(
            model, selected, receiver, directory / "sample_results.jsonl"
        )
        rows = summarize_trajectory(
            records, probes, name, expected_samples=len(selected)
        )
        write_csv(directory / "trajectory.csv", rows)
        (directory / "source.json").write_text(
            json.dumps(
                {
                    "model": name,
                    "sample_count": len(selected),
                    "seed": 42,
                    "dtype": "bfloat16",
                    "handoff": "full20[:prompt_len+k] then last k positions",
                    "receiver_position_start": "prompt_len+k",
                    "historical_runs": sources,
                    "probe_run_id": rows[0]["probe_run_id"],
                    "smoke": args.smoke,
                    "gpu": torch.cuda.get_device_name(device),
                    "git_commit": get_git_commit_hash(),
                },
                indent=2,
            )
            + "\n"
        )
        combined.extend(rows)
        del model, receiver
        gc.collect()
        torch.cuda.empty_cache()
    if models == ["0.6B", "4B", "8B"]:
        write_csv(output / "trajectory.csv", combined)
    print(f"Saved receiver trajectory: {output}")


if __name__ == "__main__":
    main()
