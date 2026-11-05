"""Explicit readouts and intervention-compatible continuation artifacts."""

import json

import torch

from ..domain.services.kv_cache import move_past_kv
from ..infrastructure.storage import save_json_atomic, save_tensor
from .intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
    forward_intervention_prefix,
)
from .intervention_continuation.identity import result_identity


def is_readout_complete(path):
    if not path.exists() or not (path.with_suffix("") / "hidden.pt").exists():
        return False
    record = json.loads(path.read_text())
    return (
        record.get("status") == "complete"
        and "rng_provenance" in record
        and bool(record.get("identity", {}).get("decoder"))
    )


def completed_steps(sample_dir, args):
    return {
        step
        for step in range(args.latent_step_start, args.latent_steps + 1)
        if all(
            is_readout_complete(
                sample_dir / "continuations" / c / f"step_{step:03d}.json"
            )
            for c in args.conditions
        )
    }


def sample_is_complete(sample_dir, args):
    for name, conditions in (("without_think", "ac"), ("with_think", "bd")):
        if set(conditions) & set(args.conditions):
            root = sample_dir / "trajectories"
            if not all(
                path.is_file()
                for path in (
                    root / f"{name}.pt",
                    root / name / "kv.pt",
                    root / name / "prompt_hidden.pt",
                )
            ):
                return False
    return (
        len(completed_steps(sample_dir, args))
        == args.latent_steps - args.latent_step_start + 1
    )


def decode_readout(
    model,
    args,
    evaluator,
    item,
    ids,
    markers,
    seed,
    index,
    opened,
    condition,
    step,
    kv,
    h,
    path,
    settings,
):
    closed = condition in ("c", "d")
    if closed:
        token = ids.new_tensor([[markers[1]]])
        branch, prefix_hidden = forward_intervention_prefix(
            model, kv, token, torch.ones_like(token)
        )
        start = prefix_hidden[-1]
    else:
        branch, start = move_past_kv(kv, model.device), h
    device = torch.device(model.device)
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda"
        else []
    )
    terminated = []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        if devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        generated, hiddens, reason, exhausted = decode_tokens_with_hiddens(
            model.model,
            model.device,
            branch,
            start.to(model.device),
            model.tokenizer,
            args.max_new_tokens,
            **settings,
            termination_ids=terminated,
            report_progress=not args.no_progress,
            progress_position=2,
            desc=f"Sample {index:06d} · {condition.upper()} · latent {step}/{args.latent_steps}",
        )
    text = model.tokenizer.decode(generated, skip_special_tokens=True).strip()
    prediction, correct, error = evaluator.evaluate(
        args.task, text, item.get("gold", "")
    )
    full_ids = generated + terminated
    record = {
        "condition": condition,
        "step": step,
        "generated_token_ids": full_ids,
        "generated_tokens": model.tokenizer.convert_ids_to_tokens(full_ids),
        "generated_text": model.tokenizer.decode(full_ids, skip_special_tokens=False),
        "evaluation_text": text,
        "prediction": prediction,
        "correct": correct,
        "gold": item.get("gold", ""),
        "error_msg": error,
        "generated_length": len(full_ids),
        "content_length": len(generated),
        "finish_reason": reason,
        "budget_exhausted": exhausted,
        "termination_token_ids": terminated,
        "inserted_prefix_ids": [markers[0]] if opened else [],
        "inserted_readout_ids": [markers[1]] if closed else [],
        "thinking_token_positions": {
            tag: [i for i, token in enumerate(full_ids) if token == marker]
            for tag, marker in zip(("<think>", "</think>"), markers, strict=True)
        },
    }
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["generated_token_ids"] != full_ids:
            raise ValueError(f"Replayed continuation differs from saved tokens: {path}")
        record = {**previous, **record}
    hidden_path = path.with_suffix("") / "hidden.pt"
    hidden = torch.stack(hiddens) if hiddens else h.new_empty((0, h.shape[-1]))
    if closed:
        hidden = (
            torch.cat([prefix_hidden, hidden[1:]]) if len(hidden) > 1 else prefix_hidden
        )
    if not hidden_path.exists():
        save_tensor(hidden_path, hidden)
    prompt_ids = ids[0].cpu().tolist()
    intervention_ids = [markers[1]] if closed else []
    record.update(
        {
            "status": "complete",
            "files": ["hidden.pt"],
            "initial_role": args.initial_role,
            "initial_prompt": model.tokenizer.decode(
                prompt_ids, skip_special_tokens=False
            ),
            "initial_prompt_token_ids": prompt_ids,
            "latent_steps_executed": step,
            "intervention_type": "control" if closed else "none",
            "intervention_prompt": "</think>" if closed else "",
            "intervention_token_ids": intervention_ids,
            "intervention_tokens": model.tokenizer.convert_ids_to_tokens(
                intervention_ids
            ),
            "intervention_length": len(intervention_ids),
            "rng_provenance": {
                "policy": "sample_v2",
                "seed": seed,
                "sample_index": index,
                "condition": condition,
            },
            "identity": result_identity(
                path.parents[4].name, step, condition, item, args, model
            ),
        }
    )
    save_json_atomic(path, record)
