"""Explicit selected-case conditional scoring, attention and latent KV intervention."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
)
from latentmas_reprop.application.intervention_continuation.identity import (
    canonical_hash,
)
from latentmas_reprop.application.latent_cache_intervention import (
    MODES,
    collect_attention,
    start_state,
)
from latentmas_reprop.application.latent_conditional_scoring import (
    answer_candidates,
    score_candidate,
)
from latentmas_reprop.application.latent_trajectory_readout import is_readout_complete
from latentmas_reprop.infrastructure.evaluators.evaluator import StandardEvaluator
from latentmas_reprop.infrastructure.models.trajectory_tokenizer import (
    load_trajectory_tokenizer,
)
from latentmas_reprop.infrastructure.storage import (
    load_json,
    load_tensor,
    save_json_atomic,
)

from .latent_causal_report import output_directory, write_summary
from .latent_trajectory_report import write_csv


@torch.inference_mode()
def analyze_case(
    model, tokenizer, directory, metadata, path, step, condition, args, output
):
    target = output / path.parent.name / f"{condition}_{step:03d}.json"
    # Stored analysis is authoritative: no latest checks, refresh or automatic rerun.
    if target.exists():
        return load_json(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    sample = load_json(path)
    record_path = path.parent / "continuations" / condition / f"step_{step:03d}.json"
    identity = {
        "source_run_id": directory.name,
        "benchmark": metadata["task"],
        "model": metadata["model_name"],
        "alignment": metadata["latent_space_realign"],
        "sample_id": sample["index"],
        "step": step,
        "condition": condition,
    }
    if not is_readout_complete(record_path):
        return {**identity, "status": "pending", "reason": "No complete saved readout"}
    record = load_json(record_path)
    name = "without_think" if condition in "ac" else "with_think"
    source = load_tensor(
        path.parent / "trajectories" / name / "kv.pt", map_location="cpu"
    )
    donor, donor_id = None, None
    for donor_path in sorted(directory.glob("samples/*/sample.json")):
        if donor_path == path:
            continue
        candidate = donor_path.parent / "trajectories" / name / "kv.pt"
        if candidate.is_file():
            donor = load_tensor(candidate, map_location="cpu")
            donor_id = load_json(donor_path)["index"]
            break
    saved_hidden = load_tensor(
        record_path.with_suffix("") / "hidden.pt", map_location="cpu"
    )
    if len(saved_hidden):
        hidden = saved_hidden[0]
    else:
        if step:
            hidden = load_tensor(
                path.parent / "trajectories" / f"{name}.pt", map_location="cpu"
            )["hidden"][step - 1]
        else:
            hidden = load_tensor(
                path.parent / "trajectories" / name / "prompt_hidden.pt",
                map_location="cpu",
            )[-1]
    records = [
        load_json(p)
        for p in sorted(path.parent.glob("continuations/*/step_*.json"))
        if is_readout_complete(p)
    ]
    prefix, candidates = answer_candidates(metadata["task"], sample, records)
    # A/B condition on an explicitly forced reasoning stop. C/D already close.
    prefix = ("\n</think>" if condition in "ab" else "") + prefix
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    region = tuple(args.region) if args.region else (0, step)
    modes = MODES if step and condition in "cd" else ("original",)
    results = []
    for mode in modes:
        if mode == "cross_sample" and donor is None:
            results.append(
                {"mode": mode, "status": "unavailable", "reason": "No same-run donor"}
            )
            continue
        row = {
            "mode": mode,
            "status": "complete",
            "donor_sample_id": donor_id if mode == "cross_sample" else None,
            "scores": [],
        }
        for candidate in candidates:
            cache, mask, h = start_state(
                model,
                source,
                step,
                condition,
                hidden,
                sample["thinking_token_ids"][1],
                mode,
                donor,
                sample["seed"],
                region,
            )
            score = score_candidate(
                model,
                cache,
                h,
                prefix_ids,
                tokenizer.encode(candidate["text"], add_special_tokens=False),
                mask,
            )
            row["scores"].append({**candidate, **score})
            del cache, mask, h
        positive = [s["string_log_probability"] for s in row["scores"] if s["correct"]]
        negative = [
            s["string_log_probability"] for s in row["scores"] if not s["correct"]
        ]
        row["candidate_margin"] = (
            max(positive) - max(negative) if positive and negative else None
        )
        if args.generate and condition in "cd":
            cache, mask, h = start_state(
                model,
                source,
                step,
                condition,
                hidden,
                sample["thinking_token_ids"][1],
                mode,
                donor,
                sample["seed"],
                region,
            )
            with torch.random.fork_rng(devices=[model.device.index]):
                torch.random.default_generator.manual_seed(sample["seed"])
                torch.cuda.manual_seed(sample["seed"])
                settings = metadata["effective_decoding"]
                tokens, _, reason, exhausted = decode_tokens_with_hiddens(
                    model,
                    model.device,
                    cache,
                    h,
                    tokenizer,
                    metadata["max_new_tokens"],
                    temperature=settings["temperature"],
                    top_p=settings["top_p"],
                    top_k=settings["top_k"],
                    eos_ids=settings["eos_ids"],
                    report_progress=False,
                    attention_mask=mask,
                )
            text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
            prediction, correct, error = StandardEvaluator().evaluate(
                metadata["task"], text, sample["gold"]
            )
            row.update(
                generated_text=text,
                prediction=prediction,
                evaluator_correct=correct,
                evaluator_error=error,
                finish_reason=reason,
                budget_exhausted=exhausted,
                generated_tokens=len(tokens),
                same_tokens_as_saved=tokens
                == record["generated_token_ids"][: record["content_length"]],
            )
            del cache, mask, h
        results.append(row)
        print(
            f"{metadata['task']} {directory.name} {sample['index']} {condition} {step} {mode}",
            flush=True,
        )
    attention = collect_attention(
        model,
        source,
        step,
        record,
        sample["thinking_token_ids"][1],
        args.attention_tokens,
    )
    write_csv(target.with_suffix(".attention.csv"), attention)
    baseline = results[0]
    for row in results:
        if (
            row.get("candidate_margin") is not None
            and baseline.get("candidate_margin") is not None
        ):
            row["margin_delta_vs_original"] = (
                row["candidate_margin"] - baseline["candidate_margin"]
            )
        if "evaluator_correct" in row and "evaluator_correct" in baseline:
            row["correctness_delta_vs_original"] = int(row["evaluator_correct"]) - int(
                baseline["evaluator_correct"]
            )
    result = {
        **identity,
        "status": "complete",
        "latent_region": list(region),
        "seed": sample["seed"],
        "generation_settings": metadata["effective_decoding"],
        "max_new_tokens": metadata["max_new_tokens"],
        "generation_executed": args.generate and condition in "cd",
        "forced_scoring_prefix": prefix,
        "scores_include_eos": False,
        "attention_queries": args.attention_tokens,
        "saved_evaluator_correct": record["correct"],
        "results": results,
    }
    save_json_atomic(target, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 40, 80])
    parser.add_argument(
        "--conditions", choices=["a", "b", "c", "d"], nargs="+", default=["c", "d"]
    )
    parser.add_argument("--region", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument("--attention-tokens", type=int, default=8)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.attention_tokens < 0:
        parser.error("--attention-tokens must be nonnegative")
    model, loaded = None, None
    for directory in args.directories:
        metadata = load_json(directory / "metadata.json")
        key = metadata["model_name"], metadata["model_revision"], metadata["dtype"]
        if key != loaded:
            del model
            model = AutoModelForCausalLM.from_pretrained(
                key[0],
                revision=key[1],
                dtype=getattr(torch, key[2].split(".")[-1]),
                local_files_only=True,
                device_map=args.device,
                attn_implementation="sdpa",
            ).eval()
            loaded = key
        tokenizer = load_trajectory_tokenizer(metadata)
        protocol = {
            "region": args.region,
            "attention_tokens": args.attention_tokens,
            "generate": args.generate,
        }
        output = output_directory(directory, metadata) / canonical_hash(protocol)[:16]
        output.mkdir(parents=True, exist_ok=True)
        cases = []
        for path in sorted(directory.glob("samples/*/sample.json")):
            for step in args.steps:
                for condition in args.conditions:
                    cases.append(
                        analyze_case(
                            model,
                            tokenizer,
                            directory,
                            metadata,
                            path,
                            step,
                            condition,
                            args,
                            output,
                        )
                    )
                    write_summary(output, cases)


if __name__ == "__main__":
    main()
