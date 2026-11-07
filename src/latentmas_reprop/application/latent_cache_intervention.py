"""Position-preserving latent KV controls for dense, full-attention Qwen3."""

import torch
from transformers.cache_utils import DynamicCache

from ..domain.services.kv_cache import dense_kv_layers

MODES = (
    "original",
    "self_transplant",
    "blocked",
    "zero_values",
    "shuffled",
    "cross_sample",
)


def rotate_half(value):
    left, right = value.chunk(2, dim=-1)
    return torch.cat([-right, left], dim=-1)


def rephase_keys(keys, source_cos, source_sin, target_cos, target_sin):
    """Undo donor RoPE and apply receiver RoPE at each destination position."""
    raw = keys.float() * source_cos + rotate_half(keys.float()) * -source_sin
    return (raw * target_cos + rotate_half(raw) * target_sin).to(keys.dtype)


def rotary_values(model, positions, device):
    if model.config.model_type != "qwen3" or model.config.sliding_window is not None:
        raise ValueError("KV intervention requires full-attention Qwen3 RoPE")
    dummy = torch.zeros(
        1, 1, model.config.hidden_size, device=device, dtype=torch.float32
    )
    cos, sin = model.model.rotary_emb(dummy, positions[None].to(device))
    return cos.float()[:, None], sin.float()[:, None]


def working_cache(
    model, source, step, mode="original", donor=None, seed=42, region=None
):
    """Create only the needed device prefix; never clone or save a complete KV.

    Keep physical length and position numbering fixed. Blocking uses a persistent
    attention mask; replacements affect only [start,end) within the latent region.
    Cross-sample and shuffled K are rephased to destination absolute positions.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown intervention: {mode}")
    prompt = source["prompt_length"]
    if not 0 <= step <= source["latent_steps"]:
        raise ValueError("Step outside saved trajectory")
    start, end = region or (0, step)
    if not 0 <= start <= end <= step:
        raise ValueError("Invalid latent intervention bounds")
    pairs, _ = dense_kv_layers(source["past_key_values"])
    device = next(model.parameters()).device
    length = prompt + step
    donor_pairs = None
    if mode == "cross_sample":
        if donor is None or donor["latent_steps"] < end:
            raise ValueError("Cross-sample donor lacks requested latent region")
        donor_pairs, _ = dense_kv_layers(donor["past_key_values"])
        if len(donor_pairs) != len(pairs):
            raise ValueError("Donor layer count differs")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(end - start, generator=generator).to(device)
    destinations = torch.arange(prompt + start, prompt + end, device=device)
    target_cos, target_sin = rotary_values(model, destinations, device)
    layers = []
    for layer, (keys, values) in enumerate(pairs):
        # .to() copies CPU source prefixes once to the device; updates never touch source.
        k = keys[..., :length, :].to(device=device, copy=True)
        v = values[..., :length, :].to(device=device, copy=True)
        region_slice = slice(prompt + start, prompt + end)
        if mode == "zero_values":
            v[..., region_slice, :] = 0
        elif mode in ("shuffled", "cross_sample", "self_transplant") and end > start:
            if mode == "cross_sample":
                dk, dv = donor_pairs[layer]
                if (
                    dk.shape[:-2] + dk.shape[-1:] != keys.shape[:-2] + keys.shape[-1:]
                    or dk.dtype != keys.dtype
                    or dv.dtype != values.dtype
                ):
                    raise ValueError("Donor KV shape or dtype differs")
                donor_prompt = donor["prompt_length"]
                donor_slice = slice(donor_prompt + start, donor_prompt + end)
                replacement_k = dk[..., donor_slice, :].to(device)
                replacement_v = dv[..., donor_slice, :].to(device)
                source_positions = torch.arange(
                    donor_prompt + start, donor_prompt + end, device=device
                )
            else:
                replacement_k, replacement_v = (
                    k[..., region_slice, :],
                    v[..., region_slice, :],
                )
                source_positions = destinations
                if mode == "shuffled":
                    replacement_k = replacement_k.index_select(-2, order)
                    replacement_v = replacement_v.index_select(-2, order)
                    source_positions = source_positions[order]
            if mode == "self_transplant":
                # Exact identity control: do not introduce avoidable FP16 roundoff.
                layers.append((k, v))
                continue
            source_cos, source_sin = rotary_values(model, source_positions, device)
            k[..., region_slice, :] = rephase_keys(
                replacement_k, source_cos, source_sin, target_cos, target_sin
            )
            v[..., region_slice, :] = replacement_v
        layers.append((k, v))
    mask = torch.ones((1, length), dtype=torch.long, device=device)
    if mode == "blocked":
        mask[:, prompt + start : prompt + end] = 0
    return DynamicCache(ddp_cache_data=layers), mask


def attention_regions(attentions, prompt, step, query_ids):
    """Summarize each query/head separately; attention is not information utility."""
    rows = []
    for layer, weights in enumerate(attentions):
        values = weights[0].float()
        for q, token in enumerate(query_ids):
            for head in range(values.shape[0]):
                p = values[head, q]
                rows.append(
                    {
                        "layer": layer,
                        "head": head,
                        "query_index": q,
                        "query_token_id": token,
                        "prompt_mass": p[:prompt].sum().item(),
                        "latent_mass": p[prompt : prompt + step].sum().item(),
                        "readout_mass": p[prompt + step :].sum().item(),
                        "total_mass": p.sum().item(),
                    }
                )
    return rows


@torch.inference_mode()
def start_state(
    model, source, step, condition, hidden, marker, mode, donor, seed, region
):
    cache, mask = working_cache(model, source, step, mode, donor, seed, region)
    if condition in "cd":
        token = torch.tensor([[marker]], device=model.device)
        mask = torch.cat([mask, torch.ones_like(token)], dim=-1)
        out = model.model(
            input_ids=token,
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        return out.past_key_values, mask, out.last_hidden_state[0, -1]
    if mode != "original":
        raise ValueError(
            "Intervention requires C/D closing-token readout; A/B start hidden would bypass changed KV"
        )
    return cache, mask, hidden.to(model.device)


@torch.inference_mode()
def collect_attention(model, source, step, record, close_id, max_queries):
    cache, mask = working_cache(model, source, step)
    query_ids = ([close_id] if record["condition"] in "cd" else []) + record[
        "generated_token_ids"
    ][:max_queries]
    if not query_ids:
        return []
    ids = torch.tensor([query_ids], device=model.device)
    mask = torch.cat([mask, torch.ones_like(ids)], dim=-1)
    model.set_attn_implementation("eager")
    try:
        out = model.model(
            input_ids=ids,
            past_key_values=cache,
            attention_mask=mask,
            output_attentions=True,
            use_cache=True,
            return_dict=True,
        )
        return attention_regions(
            out.attentions, source["prompt_length"], step, query_ids
        )
    finally:
        model.set_attn_implementation("sdpa")
