"""Exercise actual Transformers Qwen KV caches without downloading model weights."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from test_intervention_continuation import MockTokenizer
from transformers import Qwen3Config, Qwen3ForCausalLM

from latentmas_reprop.application.intervention_continuation.conditions import (
    sequential_prompts,
)
from latentmas_reprop.application.intervention_continuation.runner import (
    InterventionContinuationRunner,
)
from latentmas_reprop.application.intervention_continuation.sequential import (
    execute_sequential,
)
from latentmas_reprop.domain.services.prompts import build_initial_messages


def test_sequential_matches_independent_qwen_cache_rollout():
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=4096,
            eos_token_id=99,
            pad_token_id=0,
            attn_implementation="eager",
        )
    ).eval()

    def prepare(messages, **kwargs):
        text = "\n".join(message["content"] for message in messages[0])
        # Compact deterministic prefixes still retain role-specific token sequences.
        ids = [ord(char) % 32 for char in text[::16]]
        return (
            [text],
            torch.tensor([ids]),
            torch.ones((1, len(ids)), dtype=torch.long),
            [[str(i) for i in ids]],
        )

    wrapper = SimpleNamespace(
        model=model,
        device=torch.device("cpu"),
        tokenizer=MockTokenizer(),
        prepare_chat_batch=prepare,
        _apply_latent_realignment=lambda hidden, _: hidden,
    )
    args = SimpleNamespace(
        model_name="Qwen/tiny",
        task="gsm8k",
        initial_role="planner",
        latent_steps=1,
        max_new_tokens=1,
        temperature=0.0,
        top_p=0.95,
        top_k=20,
        intervention_conditions=["latent_intervention"],
        no_progress=True,
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)
    runner = InterventionContinuationRunner(wrapper, args, evaluator)
    item = {"question": "How much is 1+1?", "gold": "2"}
    for steps in [0, 1]:
        args.latent_steps = steps
        kv, _, _ = runner.compute_common_latent(item)
        initial_length = kv.get_seq_length()
        hidden, record = execute_sequential(runner, item, kv, steps)
        assert (
            kv.get_seq_length() == initial_length
        )  # branch must not mutate its snapshot
        # Independently prefill each role and inject the previous hidden as latent input.
        initial_ids = prepare([build_initial_messages(item["question"], args)])[1]
        role_ids = [initial_ids] + [
            torch.tensor([stage["token_ids"]])
            for stage in sequential_prompts(wrapper, item, args)
        ]
        past = None
        with torch.no_grad():
            for index, ids in enumerate(role_ids):
                output = model(
                    input_ids=ids,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = output.past_key_values
                for _ in range(steps if index < 3 else 0):
                    output = model(
                        inputs_embeds=output.hidden_states[-1][:, -1:, :],
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = output.past_key_values
        assert record["generated_token_ids"] == [int(output.logits[0, -1].argmax())]
        assert record["stages"][-1]["cache_length_after"] == past.get_seq_length()
        offset = record["generated_hidden_offset"]
        assert torch.allclose(
            hidden[offset], output.hidden_states[-1][0, -1], atol=1e-6
        )
