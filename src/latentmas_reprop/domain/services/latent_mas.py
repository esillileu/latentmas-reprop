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
        self.temperature, self.top_p = temperature, top_p
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

    def _build_messages(self, role: str, items: list[dict]) -> list[list[dict]]:
        builder = (
            build_agent_message_sequential_latent_mas
            if getattr(self.args, "prompt", "sequential") == "sequential"
            else build_agent_message_hierarchical_latent_mas
        )
        return [
            builder(
                role=role,
                question=item["question"],
                method=self.method_name,
                args=self.args,
            )
            for item in items
        ]

    def _prepare_tokens(
        self, prompts: list[str], target_device: Any
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[list[str]]]:
        wrapped = (
            [f"{p}<think>" for p in prompts]
            if getattr(self.args, "think", False)
            else prompts
        )
        encoded = self.model.tokenizer(
            wrapped, return_tensors="pt", padding=True, add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(target_device)
        mask = encoded["attention_mask"].to(target_device)
        tokens_batch = [
            self.model.tokenizer.convert_ids_to_tokens(
                ids_row[mask_row.bool()].tolist()
            )
            for ids_row, mask_row in zip(input_ids, mask, strict=False)
        ]
        return wrapped, input_ids, mask, tokens_batch

    def _evaluate_results(
        self, items: list[dict], final_texts: list[str], agent_traces: list[list[dict]]
    ) -> list[dict]:
        results = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            gold = item.get("gold", "")
            pred, ok, error_msg = self.evaluator.evaluate(self.task, final_text, gold)
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

    def _forward_non_judger(
        self, items: list[dict], record_embeddings: bool = False
    ) -> tuple[Any, list[torch.Tensor], list[list[dict]]]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_size = len(items)
        past_kv: Any = None
        agent_traces: list[list[dict]] = [[] for _ in range(batch_size)]
        embedding_record: list[torch.Tensor] = []
        target_device = self.model.HF_device if record_embeddings else self.model.device

        for agent in self.agents:
            if agent.role == "judger":
                continue
            batch_messages = self._build_messages(agent.role, items)
            prompts, _, _, _ = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )
            prev_past_len = get_past_kv_sequence_length(past_kv)
            wrapped_prompts, wrapped_ids, wrapped_mask, tokens_batch = (
                self._prepare_tokens(prompts, target_device)
            )
            if record_embeddings:
                past_kv, prev_emb = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.latent_only:
                    prev_emb = (
                        prev_emb[:, -self.latent_steps :, :]
                        if self.latent_steps > 0
                        else prev_emb[:, 0:0, :]
                    )
                embedding_record.append(prev_emb)
                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
            else:
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
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": wrapped_prompts[idx],
                        "input_ids": wrapped_ids[idx][mask].to("cpu").tolist(),
                        "input_tokens": tokens_batch[idx],
                        "latent_steps": self.latent_steps,
                        "output": "",
                    }
                )
        return past_kv, embedding_record, agent_traces

    @torch.no_grad()
    def build_latent_contexts(self, items: list[dict]) -> tuple[Any, list[list[dict]]]:
        past_kv, _, traces = self._forward_non_judger(items, record_embeddings=False)
        return past_kv, traces

    @torch.no_grad()
    def decode_with_context(
        self,
        items: list[dict],
        past_kv: Any = None,
        initial_traces: list[list[dict]] | None = None,
    ) -> list[dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_size = len(items)
        agent_traces = (
            [list(tr) for tr in initial_traces]
            if initial_traces is not None
            else [[] for _ in range(batch_size)]
        )
        judger_agent = next((a for a in self.agents if a.role == "judger"), None)
        if judger_agent is None:
            raise RuntimeError("No judger agent found in self.agents")

        batch_messages = self._build_messages(judger_agent.role, items)
        prompts, _, _, _ = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )
        past_for_decoding = past_kv if self.latent_steps > 0 else None
        judger_prompts, judger_ids, judger_mask, tokens_batch = self._prepare_tokens(
            prompts, self.model.device
        )
        generated_batch, _ = self.model.generate_text_batch(
            judger_ids,
            judger_mask,
            max_new_tokens=self.judger_max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            past_key_values=past_for_decoding,
        )
        final_texts = ["" for _ in range(batch_size)]
        for idx in range(batch_size):
            final_text = generated_batch[idx].strip()
            final_texts[idx] = final_text
            mask = judger_mask[idx].bool()
            agent_traces[idx].append(
                {
                    "name": judger_agent.name,
                    "role": judger_agent.role,
                    "input": judger_prompts[idx],
                    "input_ids": judger_ids[idx][mask].to("cpu").tolist(),
                    "input_tokens": tokens_batch[idx],
                    "output": final_text,
                }
            )
        # Results include 'error_msg' via self._evaluate_results
        return self._evaluate_results(items, final_texts, agent_traces)

    def run_batch(self, items: list[dict]) -> list[dict]:
        past_kv, agent_traces = self.build_latent_contexts(items)
        return self.decode_with_context(
            items, past_kv=past_kv, initial_traces=agent_traces
        )

    def run_batch_vllm(self, items: list[dict]) -> list[dict]:
        from .latent_mas_vllm import run_latent_mas_vllm_batch

        return run_latent_mas_vllm_batch(self, items)

    def run_item(self, item: dict) -> dict:
        return self.run_batch([item])[0]
