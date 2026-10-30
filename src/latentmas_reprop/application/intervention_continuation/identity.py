"""Identity and provenance for reusable intervention calculation states."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import transformers

from ...domain.services.prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
    build_intervention_p1_messages,
    resolve_p1_variant_id,
)
from ...infrastructure.paths.naming import clean_model_name
from ...infrastructure.storage.tensor_store import load_json, save_json_atomic


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tensor_digest(tensors: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_identity(
    model: Any, args: Any, backend: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identify loaded weights and tokenizer; reject stochastic model execution."""
    module = model.model
    if any(child.training for child in module.modules()):
        raise ValueError("Cannot reuse state with a model in training mode")
    config = module.config
    revision = getattr(config, "_commit_hash", None)
    weights = revision or _tensor_digest(sorted(module.state_dict().items()))
    tokenizer = model.tokenizer
    token_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    if not token_revision:
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if backend_tokenizer is None:
            raise ValueError("Cannot identify tokenizer contents")
        token_revision = hashlib.sha256(backend_tokenizer.to_str().encode()).hexdigest()
    realignment = bool(getattr(model, "latent_space_realign", False))
    manager = getattr(model, "realignment_manager", None)
    transform = None
    if manager is not None:
        matrix, target_norm = manager.ensure_matrix(module, torch.device(model.device))
        if realignment and not torch.is_tensor(matrix):
            raise ValueError("Cannot identify realignment matrix")
        tensors = [("target_norm", target_norm)]
        if matrix is not None:
            tensors.append(("matrix", matrix))
        transform = _tensor_digest(tensors)
    elif realignment:
        raise ValueError("Cannot identify realignment matrix")
    variant = resolve_p1_variant_id(args)
    identity = {
        "model": str(args.model_name),
        "forward_validation": "process_rng_v1",
        "weights": weights,
        "tokenizer": token_revision,
        "chat_template": tokenizer.chat_template,
        "dtype": str(next(module.parameters()).dtype),
        "backend": backend,
        "realignment": realignment,
        "realignment_digest": transform,
        "task": str(getattr(args, "task", "gsm8k")),
        "split": str(getattr(args, "split", "test")),
        "p1_variant": variant,
        "p1_messages": build_intervention_p1_messages("{question}", args),
    }
    provenance = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.device(model.device).type,
        "attention": getattr(config, "_attn_implementation", None),
    }
    return identity, provenance


def state_path(base: Path, identity: dict[str, Any]) -> Path:
    return (
        base
        / "intervention_continuation"
        / identity["task"]
        / clean_model_name(identity["model"])
        / canonical_hash(identity)
    )


def ensure_state(
    path: Path, identity: dict[str, Any], provenance: dict[str, Any]
) -> bool:
    """Record state identity; return whether extension uses the original environment."""
    path.mkdir(parents=True, exist_ok=True)
    target = path / "state.json"
    if target.exists():
        record = load_json(target)
        if record["identity"] != identity:
            raise ValueError("State identity mismatch")
        return record["provenance"] == provenance
    save_json_atomic(target, {"identity": identity, "provenance": provenance})
    return True


def result_identity(
    state_hash: str,
    step: int,
    condition: str,
    item: dict[str, Any],
    args: Any,
    model: Any,
) -> dict[str, Any]:
    payload = {
        "state": state_hash,
        "step": step,
        "condition": condition,
        "task": getattr(args, "task", "gsm8k"),
        "gold": item.get("gold", ""),
        "seed": int(getattr(args, "seed", 42)),
        "max_new_tokens": int(getattr(args, "max_new_tokens", 2048)),
        "temperature": float(getattr(args, "temperature", 0.6)),
        "top_p": float(getattr(args, "top_p", 0.95)),
        "decoder": hashlib.sha256(
            (Path(__file__).parent / "decoder.py").read_bytes()
        ).hexdigest(),
    }
    if condition == "plce":
        payload["control_signal"] = getattr(args, "control_signal", "\nSolution:")
    if condition == "plce":
        payload["intervention_token_ids"] = model.tokenizer(
            payload["control_signal"], return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].tolist()
    if condition == "plpe":
        custom = getattr(args, "plpe_prompt", None)
        if custom:
            messages = [
                {"role": "user", "content": custom.format(question=item["question"])}
            ]
        else:
            builder = (
                build_agent_message_hierarchical_latent_mas
                if getattr(args, "prompt", "sequential") == "hierarchical"
                else build_agent_message_sequential_latent_mas
            )
            messages = builder(
                role=getattr(args, "receiver_role", "judger"),
                question=item["question"],
                method="latent_mas",
                args=args,
            )
        prompts, ids, _, _ = model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        payload["intervention_prompt"] = prompts[0]
        payload["intervention_token_ids"] = ids[0].cpu().tolist()
        payload.update(
            {
                k: getattr(args, k, None)
                for k in ("plpe_prompt", "receiver_role", "prompt", "think")
            }
        )
    return payload
