"""Position controls and teacher forcing use actual model/cache semantics."""

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
)
from latentmas_reprop.application.latent_cache_intervention import (
    attention_regions,
    rephase_keys,
    rotary_values,
    working_cache,
)
from latentmas_reprop.application.latent_conditional_scoring import (
    answer_candidates,
    score_candidate,
)
from latentmas_reprop.domain.services.kv_cache import dense_kv_layers


def tiny_model():
    torch.manual_seed(7)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=31,
        )
    )
    model.eval()
    model.set_attn_implementation("eager")
    return model


def source_cache(model, prompt=3, steps=4):
    with torch.inference_mode():
        out = model.model(
            input_ids=torch.arange(prompt + steps)[None] % 32, use_cache=True
        )
    return {
        "past_key_values": out.past_key_values,
        "prompt_length": prompt,
        "latent_steps": steps,
    }


def test_transplants_keep_positions_and_source_immutable():
    model = tiny_model()
    source, donor = source_cache(model), source_cache(model, prompt=5)
    original, _ = dense_kv_layers(source["past_key_values"])
    snapshot = [(k.clone(), v.clone()) for k, v in original]
    baseline, _ = working_cache(model, source, 4)
    identity, _ = working_cache(model, source, 4, "self_transplant")
    for a, b in zip(
        dense_kv_layers(baseline)[0], dense_kv_layers(identity)[0], strict=True
    ):
        assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
    blocked, mask = working_cache(model, source, 4, "blocked", region=(1, 3))
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0, 1]]
    zero, _ = working_cache(model, source, 4, "zero_values", region=(1, 3))
    assert dense_kv_layers(zero)[0][0][1][..., 4:6, :].count_nonzero() == 0
    cross, _ = working_cache(model, source, 4, "cross_sample", donor=donor)
    k = dense_kv_layers(cross)[0][0][0]
    dk = dense_kv_layers(donor["past_key_values"])[0][0][0][..., 5:9, :]
    sc, ss = rotary_values(model, torch.arange(5, 9), torch.device("cpu"))
    tc, ts = rotary_values(model, torch.arange(3, 7), torch.device("cpu"))
    assert torch.allclose(k[..., 3:7, :], rephase_keys(dk, sc, ss, tc, ts), atol=1e-6)
    assert torch.allclose(rephase_keys(k[..., 3:7, :], tc, ts, sc, ss), dk, atol=1e-5)
    for before, after in zip(snapshot, original, strict=True):
        assert all(torch.equal(x, y) for x, y in zip(before, after, strict=True))
    assert blocked.get_seq_length() == 7


def test_conditional_score_matches_full_teacher_forcing_and_prefix_exclusion():
    model = tiny_model()
    prompt, prefix, answer = [1, 2, 3], [4, 5], [6, 7, 8]
    with torch.inference_mode():
        out = model.model(input_ids=torch.tensor([prompt]), use_cache=True)
        score = score_candidate(
            model, out.past_key_values, out.last_hidden_state[0, -1], prefix, answer
        )
        logits = (
            model(input_ids=torch.tensor([prompt + prefix + answer]), use_cache=False)
            .logits[0]
            .float()
            .log_softmax(-1)
        )
    expected = sum(
        logits[len(prompt) + len(prefix) - 1 + i, token].item()
        for i, token in enumerate(answer)
    )
    assert abs(score["string_log_probability"] - expected) < 1e-5
    assert abs(score["first_token_log_probability"] - logits[4, 6].item()) < 1e-5
    assert score["candidate_tokens"] == 3


def test_blocked_attention_stays_blocked_during_generation():
    model = tiny_model()
    source = source_cache(model)
    cache, mask = working_cache(model, source, 4, "blocked")
    observed = []

    def capture(module, args, kwargs):
        observed.append(kwargs["attention_mask"].clone())

    handle = model.model.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.inference_mode():
        token = torch.tensor([[9]])
        initial_mask = torch.cat([mask, torch.ones_like(token)], -1)
        out = model.model(
            input_ids=token,
            past_key_values=cache,
            attention_mask=initial_mask,
            output_attentions=True,
        )
        rows = attention_regions(out.attentions, 3, 4, [9])
        assert all(row["latent_mass"] == 0 for row in rows)
        assert all(abs(row["total_mass"] - 1) < 1e-6 for row in rows)
        decode_tokens_with_hiddens(
            model,
            torch.device("cpu"),
            out.past_key_values,
            out.last_hidden_state[0, -1],
            None,
            4,
            eos_ids=[],
            report_progress=False,
            attention_mask=initial_mask,
        )
    handle.remove()
    assert len(observed) == 4
    assert all(not mask[:, 3:7].any() for mask in observed)
    assert [m.shape[-1] for m in observed] == [8, 9, 10, 11]


def test_mbpp_scores_program_candidates_never_gold_tests():
    prefix, candidates = answer_candidates(
        "mbppplus",
        {"gold": "assert secret_test()"},
        [
            {"prediction": "def answer():\n    return 1", "correct": True},
            {"prediction": "bad()", "correct": False},
        ],
    )
    assert prefix.endswith("```python\n")
    assert len(candidates) == 2
    assert all("assert secret_test()" not in c["text"] for c in candidates)


def test_existing_case_is_trusted_without_loading_source(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from src.run import latent_causal_analysis as cli

    sample_path = tmp_path / "samples/000003/sample.json"
    output = tmp_path / "analysis"
    target = output / "000003/c_080.json"
    target.parent.mkdir(parents=True)
    result = {
        "status": "complete",
        "results": [{"mode": "original", "evaluator_correct": True}],
    }
    target.write_text(json.dumps(result))
    monkeypatch.setattr(
        cli,
        "load_tensor",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("stored result must not trigger tensor load")
        ),
    )
    assert (
        cli.analyze_case(
            None, None, tmp_path, {}, sample_path, 80, "c", SimpleNamespace(), output
        )
        == result
    )
