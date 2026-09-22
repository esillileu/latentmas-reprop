from typing import Any

import torch

from ...infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR, StandardEvaluator
from ...infrastructure.models.model_wrapper import ModelWrapper
from ..models import default_agents
from .kv_cache import get_past_kv_sequence_length, truncate_past_kv
from .prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


class LatentMASMethod:
    """Latent Multi-Agent System (LatentMAS) method with latent KV cache communication."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: Any = None,
        evaluator: StandardEvaluator | None = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = "latent_mas"
        self.vllm_device = getattr(args, "device", "cuda")
        self.HF_device = getattr(args, "device2", "cuda:1")
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = (
            bool(getattr(args, "sequential_info_only", False)) if args else False
        )

        if self.latent_only:
            self.sequential_info_only = True
        self.task = getattr(args, "task", "gsm8k")
        self.evaluator = evaluator or DEFAULT_EVALUATOR

    @torch.no_grad()
    def build_latent_contexts(self, items: list[dict]) -> tuple[Any, list[list[dict]]]:
        """Forward non-judger agents to construct latent communication KV cache (past_kv).

        Returns:
            tuple of (past_kv, agent_traces)
        """
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Any = None
        agent_traces: list[list[dict]] = [[] for _ in range(batch_size)]

        for agent in self.agents:
            if agent.role == "judger":
                continue

            if getattr(self.args, "prompt", "sequential") == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            else:
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            prompts, _input_ids, _attention_mask, _tokens_batch = (
                self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )
            )

            prev_past_len = get_past_kv_sequence_length(past_kv)

            if getattr(self.args, "think", False):
                wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
            else:
                wrapped_prompts = prompts

            wrapped_encoded = self.model.tokenizer(
                wrapped_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
            wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
            wrapped_tokens_batch: list[list[str]] = []
            for ids_row, mask_row in zip(wrapped_ids, wrapped_mask, strict=False):
                active_ids = ids_row[mask_row.bool()].tolist()
                wrapped_tokens_batch.append(
                    self.model.tokenizer.convert_ids_to_tokens(active_ids)
                )

            past_kv = self.model.generate_latent_batch(
                wrapped_ids,
                attention_mask=wrapped_mask,
                latent_steps=self.latent_steps,
                past_key_values=past_kv,
            )
            if self.sequential_info_only or self.latent_only:
                new_past_len = get_past_kv_sequence_length(past_kv)
                tokens_added = new_past_len - prev_past_len
                tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                past_kv = truncate_past_kv(past_kv, tokens_to_keep)

            for idx in range(batch_size):
                mask = wrapped_mask[idx].bool()
                trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": wrapped_tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    }
                )

        return past_kv, agent_traces

    @torch.no_grad()
    def decode_with_context(
        self,
        items: list[dict],
        past_kv: Any = None,
        initial_traces: list[list[dict]] | None = None,
    ) -> list[dict]:
        """Run judger agent decoding with the supplied latent communication context."""
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        agent_traces: list[list[dict]] = (
            [list(tr) for tr in initial_traces]
            if initial_traces is not None
            else [[] for _ in range(batch_size)]
        )
        final_texts = ["" for _ in range(batch_size)]

        judger_agent = next((a for a in self.agents if a.role == "judger"), None)
        if judger_agent is None:
            raise RuntimeError("No judger agent found in self.agents")

        if getattr(self.args, "prompt", "sequential") == "sequential":
            batch_messages = [
                build_agent_message_sequential_latent_mas(
                    role=judger_agent.role,
                    question=item["question"],
                    context="",
                    method=self.method_name,
                    args=self.args,
                )
                for item in items
            ]
        else:
            batch_messages = [
                build_agent_message_hierarchical_latent_mas(
                    role=judger_agent.role,
                    question=item["question"],
                    context="",
                    method=self.method_name,
                    args=self.args,
                )
                for item in items
            ]

        prompts, _input_ids, _attention_mask, _tokens_batch = (
            self.model.prepare_chat_batch(batch_messages, add_generation_prompt=True)
        )

        past_for_decoding = past_kv if self.latent_steps > 0 else None

        if getattr(self.args, "think", False):
            judger_prompts = [f"{prompt}<think>" for prompt in prompts]
        else:
            judger_prompts = prompts

        judger_encoded = self.model.tokenizer(
            judger_prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        judger_ids = judger_encoded["input_ids"].to(self.model.device)
        judger_mask = judger_encoded["attention_mask"].to(self.model.device)
        judger_tokens_batch: list[list[str]] = []
        for ids_row, mask_row in zip(judger_ids, judger_mask, strict=False):
            active_ids = ids_row[mask_row.bool()].tolist()
            judger_tokens_batch.append(
                self.model.tokenizer.convert_ids_to_tokens(active_ids)
            )

        generated_batch, _ = self.model.generate_text_batch(
            judger_ids,
            judger_mask,
            max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=past_for_decoding,
        )
        for idx in range(batch_size):
            final_text = generated_batch[idx].strip()
            final_texts[idx] = final_text
            mask = judger_mask[idx].bool()
            trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
            agent_traces[idx].append(
                {
                    "name": judger_agent.name,
                    "role": judger_agent.role,
                    "input": judger_prompts[idx],
                    "input_ids": trimmed_ids,
                    "input_tokens": judger_tokens_batch[idx],
                    "output": final_text,
                }
            )

        results: list[dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item.get("gold", "")
            pred, ok, error_msg = self.evaluator.evaluate(self.task, final_text, gold)

            if error_msg:
                print("=========================================")
                print(f"Question {idx}")
                print(f"error_msg: {error_msg}")

            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item.get("solution", ""),
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "error_msg": error_msg,
                }
            )
        return results

    @torch.no_grad()
    def run_batch(self, items: list[dict]) -> list[dict]:
        past_kv, agent_traces = self.build_latent_contexts(items)
        return self.decode_with_context(
            items, past_kv=past_kv, initial_traces=agent_traces
        )

    def run_batch_vllm(self, items: list[dict]) -> list[dict]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.judger_max_new_tokens,
        )

        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Any = None
        agent_traces: list[list[dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            if getattr(self.args, "prompt", "sequential") == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]
            elif getattr(self.args, "prompt", "sequential") == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(
                        role=agent.role,
                        question=item["question"],
                        context="",
                        method=self.method_name,
                        args=self.args,
                    )
                    for item in items
                ]

            prompts, _input_ids, _attention_mask, _tokens_batch = (
                self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )
            )

            if agent.role != "judger":
                prev_past_len = get_past_kv_sequence_length(past_kv)

                if getattr(self.args, "think", False):
                    wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(
                    self.model.HF_device
                )
                wrapped_tokens_batch: list[list[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask, strict=False):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(
                        self.model.tokenizer.convert_ids_to_tokens(active_ids)
                    )

                past_kv, previous_hidden_embedding = (
                    self.model.generate_latent_batch_hidden_state(
                        wrapped_ids,
                        attention_mask=wrapped_mask,
                        latent_steps=self.latent_steps,
                        past_key_values=past_kv,
                    )
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = get_past_kv_sequence_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = (
                        self.latent_steps if self.latent_only else tokens_added
                    )
                    past_kv = truncate_past_kv(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[
                            :, -self.latent_steps :, :
                        ]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)

                if getattr(self.args, "think", False):
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else:
                    judger_prompts = prompts

                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                curr_prompt_emb = (
                    self.model.embedding_layer(judger_encoded)
                    .squeeze(0)
                    .to(self.vllm_device)
                )

                assert (
                    "Qwen" in self.args.model_name or "qwen" in self.args.model_name
                ), (
                    "latent_embedding_position is only supported for Qwen models currently."
                )

                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)["input_ids"]))

                B, _L, H = curr_prompt_emb.shape
                _, _Lp, H = past_embedding.shape

                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat(
                        [left_emb, past_embedding[i], right_emb], dim=0
                    )
                    whole_prompt_emb_list.append(combined)

                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack(
                    [
                        torch.cat(
                            [x, torch.zeros(max_len - x.shape[0], H, device=x.device)],
                            dim=0,
                        )
                        for x in whole_prompt_emb_list
                    ]
                )

                prompt_embeds_list = [
                    {"prompt_embeds": embeds} for embeds in whole_prompt_emb
                ]

                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]

                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )

        results: list[dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item.get("gold", "")
            pred, ok, error_msg = self.evaluator.evaluate(self.task, final_text, gold)
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                    "error_msg": error_msg,
                }
            )
        return results

    def run_item(self, item: dict) -> dict:
        return self.run_batch([item])[0]
