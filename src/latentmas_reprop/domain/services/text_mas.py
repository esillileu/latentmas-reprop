from typing import Any

from ...infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR, StandardEvaluator
from ...infrastructure.models.model_wrapper import ModelWrapper
from ..models import default_agents
from .prompts import (
    build_agent_messages_hierarchical_text_mas,
    build_agent_messages_sequential_text_mas,
)
from .token_counts import attach_token_counts


class TextMASMethod:
    """Text-based Multi-Agent System (TextMAS) method."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: Any = None,
        evaluator: StandardEvaluator | None = None,
    ) -> None:
        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = getattr(args, "task", "gsm8k")
        self.evaluator = evaluator or DEFAULT_EVALUATOR

    def run_batch(self, items: list[dict]) -> list[dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: list[list[dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        token_totals = [0 for _ in range(batch_size)]

        for agent in self.agents:
            if getattr(self.args, "prompt", "sequential") == "hierarchical":
                batch_messages = [
                    build_agent_messages_hierarchical_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]
            else:
                batch_messages = [
                    build_agent_messages_sequential_text_mas(
                        role=agent.role,
                        question=item["question"],
                        context=contexts[idx],
                        method=self.method_name,
                        args=self.args,
                    )
                    for idx, item in enumerate(items)
                ]

            prompts, input_ids, attention_mask, tokens_batch = (
                self.model.prepare_chat_batch(
                    batch_messages, add_generation_prompt=True
                )
            )

            if self.model.use_vllm:
                generated_texts = self.model.vllm_generate_text_batch(
                    prompts,
                    max_new_tokens=self.max_new_tokens_each,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                generated_token_counts = [
                    len(
                        self.model.tokenizer(text, add_special_tokens=False)[
                            "input_ids"
                        ]
                    )
                    for text in generated_texts
                ]
            else:
                generated_texts, _, generated_token_counts = (
                    self.model.generate_text_batch(
                        input_ids,
                        attention_mask,
                        max_new_tokens=self.max_new_tokens_each,
                        temperature=self.temperature,
                        top_p=self.top_p,
                    )
                )

            agent_name_map_for_prompt_hierarchical = {
                "Planner": "Math Agent",
                "Critic": "Science Agent",
                "Refiner": "Code Agent",
                "Judger": "Task Summrizer",
                "planner": "Math Agent",
                "critic": "Science Agent",
                "refiner": "Code Agent",
                "judger": "Task Summrizer",
            }

            for idx in range(batch_size):
                text_out = generated_texts[idx].strip()
                token_totals[idx] += int(generated_token_counts[idx])

                if getattr(self.args, "prompt", "sequential") == "hierarchical":
                    mapped_name = agent_name_map_for_prompt_hierarchical.get(
                        agent.name, agent.name
                    )
                    formatted_output = f"[{mapped_name}]:\n{text_out}\n\n"
                else:
                    formatted_output = f"[{agent.name}]:\n{text_out}\n\n"

                if agent.role != "judger":
                    contexts[idx] = f"{contexts[idx]}{formatted_output}"
                    history_contexts[idx] = f"{history_contexts[idx]}{formatted_output}"
                else:
                    final_texts[idx] = text_out
                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                        "generated_tokens": int(generated_token_counts[idx]),
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
                attach_token_counts(
                    {
                        "question": item["question"],
                        "gold": gold,
                        "solution": item["solution"],
                        "context": history_contexts[idx],
                        "prediction": pred,
                        "raw_prediction": final_text,
                        "agents": agent_traces[idx],
                        "correct": ok,
                        "latent_steps_executed": 0,
                        "generated_tokens": token_totals[idx],
                    }
                )
            )
        return results

    def run_item(self, item: dict) -> dict:
        return self.run_batch([item])[0]
