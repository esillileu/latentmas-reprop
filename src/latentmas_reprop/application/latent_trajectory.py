"""Collect two shared latent trajectories and independent explicit readouts."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

from ..domain.services.kv_cache import (
    get_past_kv_sequence_length,
    slice_past_kv,
)
from ..domain.services.prompts import build_initial_messages
from ..infrastructure.storage import load_tensor, save_json_atomic, save_tensor
from .intervention_continuation.decoder import (
    generation_settings,
    get_lm_head,
)
from .intervention_continuation.identity import sample_seed
from .intervention_continuation.runner import InterventionContinuationRunner
from .latent_trajectory_readout import (
    completed_steps,
    decode_readout,
    is_readout_complete,
    sample_is_complete,
)
from .latent_trajectory_samples import select_samples


def thinking_ids(tokenizer):
    ids = [tokenizer.convert_tokens_to_ids(t) for t in ("<think>", "</think>")]
    if any(i is None or i == tokenizer.unk_token_id for i in ids) or len(set(ids)) != 2:
        raise ValueError("Tokenizer must provide distinct thinking tokens")
    for tag, token_id in zip(("<think>", "</think>"), ids, strict=True):
        if tokenizer.encode(tag, add_special_tokens=False) != [token_id]:
            raise ValueError("Thinking markers must encode to native single tokens")
    return ids


def prepare_prompt(model, messages, markers):
    """Remove only the template's known thinking suffix after the assistant header."""
    prompts, ids, _, _ = model.prepare_chat_batch(
        [messages],
        add_generation_prompt=True,
        chat_template_kwargs={"enable_thinking": True},
    )
    base = model.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
    )
    text = prompts[0]
    if not text.startswith(base):
        raise ValueError("Chat template must append an assistant generation prefix")
    suffix = text[len(base) :]
    if "<think>" in suffix:
        header, tail = suffix.split("<think>", 1)
        if tail.strip():
            raise ValueError("Unexpected automatic thinking prefix")
        text = base + header
        ids = model.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ].to(model.device)
    suffix_ids = model.tokenizer(text[len(base) :], add_special_tokens=False)[
        "input_ids"
    ]
    if any(i in markers for i in suffix_ids):
        raise ValueError("Assistant prefix contains thinking markers")
    return text, ids


def ensure_trajectory(model, args, runner, item, ids, trajectory_path):
    """Persist one complete prompt+latent cache before any explicit readouts."""
    kv_path = trajectory_path.with_suffix("") / "kv.pt"
    prompt_hidden_path = trajectory_path.with_suffix("") / "prompt_hidden.pt"
    if trajectory_path.exists() and kv_path.exists() and prompt_hidden_path.exists():
        return
    output = model.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    if not prompt_hidden_path.exists():
        save_tensor(prompt_hidden_path, output.hidden_states[-1][0].detach().cpu())
    if trajectory_path.exists() and kv_path.exists():
        return
    initial = {
        **item,
        "prompt": model.tokenizer.decode(ids[0]),
        "prompt_token_ids": ids[0].tolist(),
        "prompt_length": ids.shape[1],
    }
    kv, hiddens, _ = runner.compute_common_latent(
        initial,
        initial_kv=output.past_key_values,
        initial_hiddens=output.hidden_states[-1][0].detach().cpu(),
    )
    hidden = hiddens[ids.shape[1] :]
    data = {
        "steps": torch.arange(1, args.latent_steps + 1),
        "hidden": hidden,
        "logits": torch.stack(
            [get_lm_head(model.model)(h.to(model.device)).float().cpu() for h in hidden]
        ),
        "prompt_token_ids": ids[0].cpu(),
    }
    if trajectory_path.exists():
        previous = torch.load(trajectory_path, weights_only=True)
        if any(not torch.equal(data[key], previous[key]) for key in data):
            raise ValueError("Recomputed trajectory differs from saved tensors")
    else:
        save_tensor(trajectory_path, data)
    if not kv_path.exists():
        save_tensor(
            kv_path,
            {
                "past_key_values": kv,
                "prompt_length": ids.shape[1],
                "latent_steps": args.latent_steps,
            },
        )


@torch.inference_mode()
def collect(model, args, dataset_port, evaluator, directory, sample_context=None):
    directory = Path(directory)
    markers = thinking_ids(model.tokenizer)
    settings = generation_settings(model, args)
    dataset = dataset_port.load(task=args.task, split=args.split)
    runner_args = SimpleNamespace(**vars(args))
    runner_args.latent_step_checkpoint_interval = 0
    runner = InterventionContinuationRunner(model, runner_args, evaluator)
    samples = list(select_samples(dataset, args))
    finished_samples = {
        index
        for index, _ in samples
        if sample_is_complete(directory / "samples" / f"{index:06d}", args)
    }
    count = 0
    alignment = "ON" if args.latent_space_realign else "OFF"
    with tqdm(
        total=len(samples),
        initial=len(finished_samples),
        unit="sample",
        position=0,
        dynamic_ncols=True,
        desc=f"Samples · alignment={alignment}",
        disable=args.no_progress,
    ) as sample_progress:
        for index, item in samples:
            sample_dir = directory / "samples" / f"{index:06d}"
            context = (
                sample_context(sample_dir)
                if sample_context is not None
                else nullcontext(None)
            )
            with context as step_completed:
                messages = build_initial_messages(item["question"], args)
                prompt, prompt_ids = prepare_prompt(model, messages, markers)
                seed = sample_seed(
                    args.seed, args.task, args.split, index, item["question"]
                )
                save_json_atomic(
                    sample_dir / "sample.json",
                    {
                        "index": index,
                        **item,
                        "messages": messages,
                        "prompt": prompt,
                        "prompt_token_ids": prompt_ids[0].tolist(),
                        "seed": seed,
                        "thinking_token_ids": markers,
                        "initial_role": args.initial_role,
                        "initial_states": {
                            name: {
                                "prompt": prompt + ("<think>" if opened else ""),
                                "prompt_token_ids": prompt_ids[0].tolist()
                                + ([markers[0]] if opened else []),
                                "prompt_length": prompt_ids.shape[1] + int(opened),
                                "latent_steps": args.latent_steps,
                                "kv_path": f"trajectories/{name}/kv.pt",
                                "hidden_path": f"trajectories/{name}.pt",
                                "prompt_hidden_path": f"trajectories/{name}/prompt_hidden.pt",
                            }
                            for opened, name in (
                                (False, "without_think"),
                                (True, "with_think"),
                            )
                        },
                    },
                )
                trajectories = []
                for opened, pair in ((False, ("a", "c")), (True, ("b", "d"))):
                    selected = [c for c in pair if c in args.conditions]
                    if not selected:
                        continue
                    ids = prompt_ids
                    if opened:
                        ids = torch.cat([ids, ids.new_tensor([[markers[0]]])], dim=1)
                    path = (
                        sample_dir
                        / "trajectories"
                        / f"{'with' if opened else 'without'}_think.pt"
                    )
                    ensure_trajectory(model, args, runner, item, ids, path)
                    trajectories.append((opened, selected, ids, path))

                # Interleave the four readouts by step from the durable trajectories.
                branches = {}
                for opened, selected, ids, trajectory_path in trajectories:
                    data = torch.load(trajectory_path, weights_only=True)
                    snapshot = load_tensor(
                        trajectory_path.with_suffix("") / "kv.pt", map_location="cpu"
                    )
                    prompt_length = snapshot["prompt_length"]
                    if (
                        prompt_length != ids.shape[1]
                        or snapshot["latent_steps"] != args.latent_steps
                        or get_past_kv_sequence_length(snapshot["past_key_values"])
                        != prompt_length + args.latent_steps
                        or not torch.equal(data["prompt_token_ids"], ids[0].cpu())
                    ):
                        raise ValueError(
                            "Saved trajectory cache does not match the requested prefix"
                        )
                    for condition in selected:
                        branches[condition] = (opened, ids, data, snapshot)
                total = args.latent_steps - args.latent_step_start + 1
                completed = completed_steps(sample_dir, args)
                with tqdm(
                    total=total,
                    initial=len(completed),
                    unit="step",
                    position=1,
                    dynamic_ncols=True,
                    desc=f"Sample {index:06d}",
                    disable=args.no_progress,
                ) as progress:
                    for step in range(args.latent_step_start, args.latent_steps + 1):
                        progress.set_description(
                            f"Sample {index:06d} · step {step}/{args.latent_steps}",
                            refresh=False,
                        )
                        for condition in (c for c in "abcd" if c in branches):
                            path = (
                                sample_dir
                                / "continuations"
                                / condition
                                / f"step_{step:03d}.json"
                            )
                            if is_readout_complete(path):
                                continue
                            opened, ids, data, snapshot = branches[condition]
                            kv = slice_past_kv(
                                snapshot["past_key_values"],
                                0,
                                snapshot["prompt_length"] + step,
                            )
                            decode_readout(
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
                                data["hidden"][step - 1],
                                path,
                                settings,
                            )
                        if step_completed is not None:
                            step_completed(step)
                        if step not in completed:
                            progress.update(1)
            if index not in finished_samples:
                sample_progress.update(1)
            count += 1
    return count
