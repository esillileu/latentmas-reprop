from typing import Any

from ...infrastructure.evaluators.evaluator import DEFAULT_EVALUATOR, StandardEvaluator
from ...infrastructure.models.model_wrapper import ModelWrapper
from .prompts import build_agent_messages_single_agent
from .token_counts import attach_token_counts


class BaselineMethod:
    """Baseline single-agent generation method."""

    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args: Any = None,
        evaluator: StandardEvaluator | None = None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = getattr(args, "task", "gsm8k")
        self.evaluator = evaluator or DEFAULT_EVALUATOR

    def run_batch(self, items: list[dict]) -> list[dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_messages = [
            build_agent_messages_single_agent(question=item["question"], args=self.args)
            for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = (
            self.model.prepare_chat_batch(batch_messages, add_generation_prompt=True)
        )

        if self.use_vllm:
            generated_batch = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            generated_token_counts = [
                len(self.model.tokenizer(text, add_special_tokens=False)["input_ids"])
                for text in generated_batch
            ]
        else:
            generated_batch, _, generated_token_counts = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        results: list[dict] = []

        for idx, item in enumerate(items):
            generated_text = generated_batch[idx]
            gold = item.get("gold", "")
            pred, ok, error_msg = self.evaluator.evaluate(
                self.task, generated_text, gold
            )

            if error_msg:
                print("=========================================")
                print(f"Question {idx}")
                print(f"error_msg: {error_msg}")

            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
                "generated_tokens": int(generated_token_counts[idx]),
            }
            results.append(
                attach_token_counts(
                    {
                        "question": item["question"],
                        "gold": gold,
                        "solution": item["solution"],
                        "prediction": pred,
                        "raw_prediction": generated_text,
                        "agents": [agent_trace],
                        "correct": ok,
                        "latent_steps_executed": 0,
                    }
                )
            )
        return results

    def run_item(self, item: dict) -> dict:
        return self.run_batch([item])[0]
