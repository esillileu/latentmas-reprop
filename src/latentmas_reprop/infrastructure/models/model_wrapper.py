import contextlib
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...domain.ports.cache_port import CacheLayer
from ...domain.ports.model_port import LatentRolloutState, ModelPort, NextTokenState
from ...domain.services.kv_cache import get_past_kv_sequence_length
from ..cache.manager import ExecutionCacheManager, get_cache_manager

try:
    from vllm import LLM, SamplingParams

    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})


class ModelWrapper(ModelPort):
    """Model adapter supporting HuggingFace Transformers and vLLM backends.

    Features:
    - Implements ModelPort interface
    - Caches latent realignment matrices into root .cache/models/realign/
    - Fixes transformers 5.x generate kwargs compatibility
    - Uses dtype instead of deprecated torch_dtype
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device | str,
        use_vllm: bool = False,
        args: Any = None,
        cache_manager: ExecutionCacheManager | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_vllm = use_vllm and _HAS_VLLM
        self.vllm_engine = None
        self.latent_space_realign = (
            bool(getattr(args, "latent_space_realign", False)) if args else False
        )
        self._latent_realign_matrices: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args
        self.cache_manager = cache_manager or get_cache_manager()

        # For ablation inspection
        self.pre_aligned = None

        if self.use_vllm:
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))

            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if (
                getattr(args, "enable_prefix_caching", False)
                and getattr(args, "method", "") == "latent_mas"
            ):
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                    enable_prefix_caching=True,
                    enable_prompt_embeds=True,
                )
            else:
                self.vllm_engine = LLM(
                    model=model_name,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=gpu_util,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

            use_second_hf = (
                bool(getattr(args, "use_second_HF_model", False)) if args else False
            )
            if use_second_hf:
                self.HF_model = (
                    AutoModelForCausalLM.from_pretrained(
                        model_name,
                        dtype=(
                            torch.bfloat16
                            if torch.cuda.is_available()
                            else torch.float32
                        ),
                    )
                    .to(args.device2)
                    .eval()
                )
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                self._ensure_latent_realign_matrix(
                    self.HF_model, torch.device(self.HF_device), args
                )
            elif self.latent_space_realign:
                raise ValueError(
                    "latent_space_realign requires --use_second_HF_model when using vLLM backend."
                )
            _ensure_pad_token(self.tokenizer)
            return

        # Fallback: HuggingFace Transformers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **(chat_template_kwargs or {}),
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> tuple[str, torch.Tensor, torch.Tensor, list[str]]:
        prompt_text = self.render_chat(
            messages, add_generation_prompt=add_generation_prompt
        )
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: list[list[dict]],
        add_generation_prompt: bool = True,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, list[list[str]]]:
        prompts: list[str] = []
        for messages in batch_messages:
            prompts.append(
                self.render_chat(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    chat_template_kwargs=chat_template_kwargs,
                )
            )
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: list[list[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask, strict=False):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        if not self.vllm_engine:
            raise RuntimeError(
                "vLLM engine not initialized. Pass use_vllm=True to ModelWrapper."
            )
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations

    def _build_latent_realign_matrix(
        self, model: torch.nn.Module, device: torch.device, args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = (
            model.get_input_embeddings()
            if hasattr(model, "get_input_embeddings")
            else None
        )
        output_embeds = (
            model.get_output_embeddings()
            if hasattr(model, "get_output_embeddings")
            else None
        )
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError(
                "Cannot build latent realignment matrix: embedding weights not accessible."
            )
        input_weight = input_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        output_weight = output_embeds.weight.detach().to(
            device=device, dtype=torch.float32
        )
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if getattr(self.args, "latent_space_realign", False):
            pass
        else:
            # Identity matrix fallback for normalization
            realign_matrix = torch.eye(
                realign_matrix.shape[0],
                device=realign_matrix.device,
                dtype=realign_matrix.dtype,
            )

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(
        self, model: torch.nn.Module, device: torch.device, args: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is not None:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)
                target_norm = target_norm.to(target_device)
                self._latent_realign_matrices[key] = (matrix, target_norm)
            return matrix, target_norm

        # Check persistent execution cache at .cache/models/realign/
        sanitized_model = self.model_name.replace("/", "_").replace("\\", "_")
        realign_flag = (
            "realigned" if getattr(args, "latent_space_realign", False) else "identity"
        )
        cache_filename = f"{sanitized_model}_{realign_flag}_realign.pt"

        if self.cache_manager.exists(CacheLayer.MODELS_REALIGN, cache_filename):
            try:
                cached_data = self.cache_manager.load_torch(
                    CacheLayer.MODELS_REALIGN,
                    cache_filename,
                    map_location=target_device,
                )
                matrix = cached_data["matrix"].to(target_device)
                target_norm = cached_data["target_norm"].to(target_device)
                self._latent_realign_matrices[key] = (matrix, target_norm)
                return matrix, target_norm
            except Exception:
                pass

        # Compute matrix
        matrix, target_norm = self._build_latent_realign_matrix(
            model, target_device, args
        )

        target_norm = (
            target_norm.to(device=target_device, dtype=matrix.dtype)
            if isinstance(target_norm, torch.Tensor)
            else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        )

        # Save to persistent execution cache
        with contextlib.suppress(Exception):
            self.cache_manager.save_torch(
                CacheLayer.MODELS_REALIGN,
                cache_filename,
                {"matrix": matrix.cpu(), "target_norm": target_norm.cpu()},
            )

        self._latent_realign_matrices[key] = (matrix, target_norm)
        return matrix, target_norm

    def _apply_latent_realignment(
        self, hidden: torch.Tensor, model: torch.nn.Module
    ) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(
            model, hidden.device, self.args
        )
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Any = None,
    ) -> tuple[list[str], Any]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()

        if past_key_values is not None:
            past_len = get_past_kv_sequence_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=False,
            past_key_values=past_key_values,
        )
        sequences = outputs.sequences
        generations: list[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    @torch.no_grad()
    def forward_next_token_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        past_key_values: Any = None,
        output_hidden_states: bool = False,
        position_start: int | None = None,
    ) -> NextTokenState:
        if self.use_vllm:
            raise RuntimeError(
                "Next-token state inspection requires the transformers backend"
            )
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        if past_key_values is not None:
            past_len = get_past_kv_sequence_length(past_key_values)
            if past_len:
                prefix = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat((prefix, attention_mask), dim=-1)
        position_ids = None
        if position_start is not None:
            position_ids = (
                torch.arange(
                    position_start,
                    position_start + input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                )
                .unsqueeze(0)
                .expand(input_ids.shape[0], -1)
            )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = None
        if output_hidden_states:
            hidden = tuple(state[:, -1, :].detach() for state in outputs.hidden_states)
        return NextTokenState(
            logits=outputs.logits[:, -1, :].detach(), hidden_states=hidden
        )

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> Any:
        return self._generate_latent_rollout(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        ).past_key_values

    @torch.no_grad()
    def generate_latent_batch_with_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentRolloutState:
        return self._generate_latent_rollout(
            input_ids,
            attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
        )

    def _generate_latent_rollout(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> LatentRolloutState:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = get_past_kv_sequence_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        last_hidden = outputs.hidden_states[-1][:, -1, :]
        hidden_steps: list[torch.Tensor] = []
        latent_steps_output: list[torch.Tensor] = []

        for _ in range(latent_steps):
            hidden_steps.append(last_hidden.detach())
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_steps_output.append(latent_vec.detach())
            latent_embed = latent_vec.unsqueeze(1)

            past_len = get_past_kv_sequence_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            outputs = self.model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        batch_size = input_ids.shape[0]
        hidden_size = last_hidden.shape[-1]
        empty = last_hidden.new_empty((batch_size, 0, hidden_size))
        return LatentRolloutState(
            past_key_values=past,
            hidden_pre_realign=torch.stack(hidden_steps, dim=1)
            if hidden_steps
            else empty,
            latent_post_realign=torch.stack(latent_steps_output, dim=1)
            if latent_steps_output
            else empty.clone(),
        )

    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        latent_steps: int,
        past_key_values: Any = None,
    ) -> tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = get_past_kv_sequence_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        curr_output_embedding = []
        curr_output_embedding.append(outputs.hidden_states[0])

        for _ in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = get_past_kv_sequence_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1)
