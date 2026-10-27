"""Execution logic for Common Latent computation and PLE/PLCE/PLPE branching."""

from typing import Any

import torch
from tqdm import tqdm

from ...domain.services.kv_cache import (
    clone_past_kv,
    get_past_kv_sequence_length,
    move_past_kv,
)
from ...domain.services.prompts import (
    build_agent_message_hierarchical_latent_mas,
    build_agent_message_sequential_latent_mas,
)
from .decoder import forward_intervention_prefix, run_explicit_generation


class SingleInterventionRunner:
    """Executes UC-01 common latent calculation and UC-02 three branch conditions."""

    def __init__(self, model: Any, args: Any, evaluator: Any) -> None:
        self.model = model
        self.args = args
        self.evaluator = evaluator

    @torch.no_grad()
    def compute_common_latent(
        self, item: dict[str, Any]
    ) -> tuple[Any, torch.Tensor, dict[str, Any]]:
        """UC-01: P_1 -> L_1 -> ... -> L_T common latent calculation."""
        role = getattr(self.args, "sender_role", "planner")
        messages = build_agent_message_sequential_latent_mas(
            role=role, question=item["question"], method="latent_mas", args=self.args
        )
        prompts, ids, mask, tokens = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        outputs = self.model.model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        prompt_hidden = outputs.hidden_states[-1][0].detach().cpu()
        last_hidden = prompt_hidden[-1:, :].to(self.model.device)

        latent_steps = int(getattr(self.args, "latent_steps", 10))
        latent_hiddens: list[torch.Tensor] = []
        show_prog = not bool(getattr(self.args, "no_progress", False))
        latent_iter = tqdm(
            range(latent_steps),
            desc="Common latent steps",
            unit="step",
            leave=False,
            disable=not show_prog,
        )
        for _ in latent_iter:
            latent_vec = self.model._apply_latent_realignment(
                last_hidden, self.model.model
            )
            latent_embed = latent_vec.unsqueeze(1)
            past_len = get_past_kv_sequence_length(past)
            latent_mask = torch.ones(
                (1, past_len + 1), dtype=torch.long, device=self.model.device
            )
            step_outputs = self.model.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = step_outputs.past_key_values
            step_hidden = step_outputs.hidden_states[-1][0, -1, :]
            latent_hiddens.append(step_hidden.detach().cpu())
            last_hidden = step_hidden.unsqueeze(0)

        latent_stack = (
            torch.stack(latent_hiddens, dim=0)
            if latent_hiddens
            else prompt_hidden.new_empty((0, prompt_hidden.shape[-1]))
        )
        common_hidden = torch.cat([prompt_hidden, latent_stack], dim=0)
        kv_snapshot = move_past_kv(clone_past_kv(past), "cpu")
        meta = {
            "prompt": prompts[0],
            "prompt_token_ids": ids[0].cpu().tolist(),
            "prompt_tokens": tokens[0],
            "prompt_length": int(prompt_hidden.shape[0]),
            "latent_steps": latent_steps,
        }
        return kv_snapshot, common_hidden, meta

    @torch.no_grad()
    def execute_ple(
        self, item: dict[str, Any], kv_snapshot: Any, last_hidden: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """UC-02 condition ple: P_1 -> L -> E (no additional intervention input)."""
        kv = move_past_kv(clone_past_kv(kv_snapshot), self.model.device)
        ids, hiddens, reason, exh, text = run_explicit_generation(
            self.model, kv, last_hidden, self.args, desc="Decoding · PLE"
        )
        pred, ok, err = self.evaluator.evaluate(
            getattr(self.args, "task", "gsm8k"), text, item.get("gold", "")
        )
        hidden_tensor = (
            torch.stack(hiddens, dim=0)
            if hiddens
            else last_hidden.new_empty((0, last_hidden.shape[-1]))
        )
        return hidden_tensor, {
            "condition": "ple",
            "intervention_type": "none",
            "intervention_length": 0,
            "intervention_token_ids": [],
            "intervention_tokens": [],
            "generated_text": text,
            "generated_token_ids": ids,
            "generated_tokens": self.model.tokenizer.convert_ids_to_tokens(ids),
            "generated_length": len(ids),
            "finish_reason": reason,
            "budget_exhausted": exh,
            "prediction": pred,
            "gold": item.get("gold", ""),
            "correct": ok,
            "error_msg": err,
        }

    @torch.no_grad()
    def execute_plce(
        self, item: dict[str, Any], kv_snapshot: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """UC-02 condition plce: P_1 -> L -> C -> E (control signal intervention)."""
        sig = str(getattr(self.args, "control_signal", "\nSolution:"))
        enc = self.model.tokenizer(sig, return_tensors="pt", add_special_tokens=False)
        c_ids, c_mask = enc["input_ids"], enc["attention_mask"]
        kv, c_hidden = forward_intervention_prefix(
            self.model, kv_snapshot, c_ids, c_mask
        )
        ids, hiddens, reason, exh, text = run_explicit_generation(
            self.model, kv, c_hidden[-1], self.args, desc="Decoding · PLCE"
        )
        pred, ok, err = self.evaluator.evaluate(
            getattr(self.args, "task", "gsm8k"), text, item.get("gold", "")
        )
        gen_stack = (
            torch.stack(hiddens, dim=0)
            if hiddens
            else c_hidden.new_empty((0, c_hidden.shape[-1]))
        )
        full_hidden = (
            torch.cat([c_hidden, gen_stack[1:]], dim=0)
            if len(gen_stack) > 1
            else c_hidden
        )
        return full_hidden, {
            "condition": "plce",
            "intervention_type": "control",
            "intervention_signal": sig,
            "intervention_token_ids": c_ids[0].cpu().tolist(),
            "intervention_tokens": self.model.tokenizer.convert_ids_to_tokens(
                c_ids[0].cpu().tolist()
            ),
            "intervention_length": int(c_ids.shape[1]),
            "generated_text": text,
            "generated_token_ids": ids,
            "generated_tokens": self.model.tokenizer.convert_ids_to_tokens(ids),
            "generated_length": len(ids),
            "finish_reason": reason,
            "budget_exhausted": exh,
            "prediction": pred,
            "gold": item.get("gold", ""),
            "correct": ok,
            "error_msg": err,
        }

    @torch.no_grad()
    def execute_plpe(
        self, item: dict[str, Any], kv_snapshot: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """UC-02 condition plpe: P_1 -> L -> P_2 -> E (role/problem prompt intervention)."""
        custom_p = getattr(self.args, "plpe_prompt", None)
        if custom_p:
            q_text = custom_p.format(question=item["question"])
            messages = [{"role": "user", "content": q_text}]
        else:
            role = getattr(self.args, "receiver_role", "judger")
            prompt_style = getattr(self.args, "prompt", "sequential")
            builder = (
                build_agent_message_hierarchical_latent_mas
                if prompt_style == "hierarchical"
                else build_agent_message_sequential_latent_mas
            )
            messages = builder(
                role=role, question=item["question"], method="latent_mas", args=self.args
            )
        prompts, p2_ids, p2_mask, p2_tokens = self.model.prepare_chat_batch(
            [messages], add_generation_prompt=True
        )
        kv, p2_hidden = forward_intervention_prefix(
            self.model, kv_snapshot, p2_ids, p2_mask
        )
        ids, hiddens, reason, exh, text = run_explicit_generation(
            self.model, kv, p2_hidden[-1], self.args, desc="Decoding · PLPE"
        )
        pred, ok, err = self.evaluator.evaluate(
            getattr(self.args, "task", "gsm8k"), text, item.get("gold", "")
        )
        gen_stack = (
            torch.stack(hiddens, dim=0)
            if hiddens
            else p2_hidden.new_empty((0, p2_hidden.shape[-1]))
        )
        full_hidden = (
            torch.cat([p2_hidden, gen_stack[1:]], dim=0)
            if len(gen_stack) > 1
            else p2_hidden
        )
        return full_hidden, {
            "condition": "plpe",
            "intervention_type": "prompt",
            "intervention_prompt": prompts[0],
            "intervention_token_ids": p2_ids[0].cpu().tolist(),
            "intervention_tokens": p2_tokens[0],
            "intervention_length": int(p2_ids.shape[1]),
            "generated_text": text,
            "generated_token_ids": ids,
            "generated_tokens": self.model.tokenizer.convert_ids_to_tokens(ids),
            "generated_length": len(ids),
            "finish_reason": reason,
            "budget_exhausted": exh,
            "prediction": pred,
            "gold": item.get("gold", ""),
            "correct": ok,
            "error_msg": err,
        }
