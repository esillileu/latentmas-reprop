"""Explicitly analyze saved readout starts and injected vectors; no generation."""

import argparse
from pathlib import Path

import torch

from latentmas_reprop.application.latent_trajectory_readout import is_readout_complete
from latentmas_reprop.application.latent_weight_analysis import (
    distribution_features,
    embedding_comparison,
    injected_vectors,
    load_alignment,
    load_embedding_weights,
)
from latentmas_reprop.infrastructure.storage import (
    load_json,
    load_tensor,
    save_json_atomic,
)

from .latent_trajectory_report import write_csv


@torch.inference_mode()
def analyze(directory, embedding, head, output, realign_root):
    metadata = load_json(directory / "metadata.json")
    mode = "realigned" if metadata["latent_space_realign"] else "identity"
    matrix, target = load_alignment(
        metadata,
        realign_root / f"{metadata['model_name'].replace('/', '_')}_{mode}_realign.pt",
        head.device,
    )
    starts, injections, origins = [], [], []
    for path in sorted(directory.glob("samples/*/sample.json")):
        sample = load_json(path)
        identity = {
            "source_run_id": directory.name,
            "benchmark": metadata["task"],
            "model": metadata["model_name"],
            "alignment": metadata["latent_space_realign"],
            "sample_id": sample["index"],
        }
        states = {}
        for name in ("without_think", "with_think"):
            root = path.parent / "trajectories"
            if not (root / f"{name}.pt").is_file():
                continue
            data = load_tensor(root / f"{name}.pt", map_location="cpu")
            prompt = load_tensor(root / name / "prompt_hidden.pt", map_location="cpu")
            hidden = torch.cat([prompt[-1:], data["hidden"]]).to(head.device)
            states[name] = hidden
            origins.append(
                {
                    **identity,
                    "trajectory_type": name,
                    "step": 0,
                    "context": "prompt_raw",
                    **distribution_features(
                        hidden[0], head, sample["thinking_token_ids"]
                    ),
                }
            )
            vectors = injected_vectors(hidden[:-1], matrix, target)
            comparisons = embedding_comparison(vectors, embedding)
            injections.extend(
                {
                    **identity,
                    "trajectory_type": name,
                    "step": i + 1,
                    "input_hidden_step": i,
                    **row,
                }
                for i, row in enumerate(comparisons)
            )
        for condition in "abcd":
            name = "without_think" if condition in "ac" else "with_think"
            for step in range(
                metadata["latent_step_start"], metadata["latent_steps"] + 1
            ):
                record = (
                    path.parent / "continuations" / condition / f"step_{step:03d}.json"
                )
                row = {
                    **identity,
                    "condition": condition,
                    "step": step,
                    "context": "after_closing_prefill"
                    if condition in "cd"
                    else "shared_trajectory_raw",
                    "status": "missing",
                }
                if is_readout_complete(record):
                    saved = load_tensor(
                        record.with_suffix("") / "hidden.pt", map_location="cpu"
                    )
                    # Decoder persists the state predicting the first generated token.
                    # C/D always persist closing-prefill hidden even for immediate EOS.
                    h = (
                        saved[0]
                        if len(saved)
                        else states.get(name, [])[step]
                        if condition in "ab" and name in states
                        else None
                    )
                    if h is not None:
                        row.update(
                            distribution_features(
                                h, head, sample["thinking_token_ids"]
                            ),
                            status="complete",
                        )
                    else:
                        row["status"] = "unavailable"
                starts.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "trajectory_origin_logits.csv", origins)
    write_csv(output / "readout_start_logits.csv", starts)
    write_csv(output / "injected_vectors.csv", injections)
    save_json_atomic(
        output / "coverage.json",
        {
            "source": str(directory.resolve()),
            "trajectory_origins": len(origins),
            "start_distributions": len(starts),
            "available_start_distributions": sum(
                r["status"] == "complete" for r in starts
            ),
            "injected_vectors": len(injections),
            "model_forward": False,
        },
    )
    (output / "summary.md").write_text(
        "# Recorded-weight analysis\n\nC4: raw, untempered LM-head distributions of the actual saved readout-start states; C/D are after closing-prefill. Missing readouts stay missing.\n\nD4: injection step k uses output hidden k-1, then the recorded alignment matrix and norm matching, cast to the recorded inference dtype. Nearest embedding cosine is a distribution diagnostic, not reasoning evidence.\n\nStep 0 prompt-state projections are independent of readout availability: [trajectory origins](trajectory_origin_logits.csv).\n\nTables: [start logits](readout_start_logits.csv), [injected vectors](injected_vectors.csv), [coverage](coverage.json). No generation or source modification.\n"
    )
    print(f"Weight analysis: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--realign-root", type=Path, default=Path(".cache/models/realign")
    )
    args = parser.parse_args(argv)
    loaded, embedding, head = None, None, None
    for directory in args.directories:
        metadata = load_json(directory / "metadata.json")
        key = metadata["model_name"], metadata["model_revision"], metadata["dtype"]
        if key != loaded:
            del embedding, head
            embedding, head = load_embedding_weights(metadata, args.device)
            loaded = key
        output = (
            Path("artifacts/latent_trajectory")
            / metadata["task"]
            / directory.parent.name
            / directory.name
            / "weight_analysis"
        )
        analyze(directory, embedding, head, output, args.realign_root)


if __name__ == "__main__":
    main()
