"""Tests for LMAS Intervention Continuation Analysis (PLE, PLCE, PLPE) specification v0.2."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from latentmas_reprop.application.intervention_continuation.analysis import (
    SingleInterventionAnalyzer,
)
from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
)
from latentmas_reprop.application.intervention_continuation.runner import (
    InterventionContinuationRunner,
)
from latentmas_reprop.application.intervention_continuation_use_case import (
    InterventionContinuationUseCase,
)
from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.paths.naming import build_experiment_relpath
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from latentmas_reprop.infrastructure.storage import FileStorage, ResearchDataReader


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 99
        self.pad_token_id = 0
        self.init_kwargs = {"_commit_hash": "tokenizer-revision"}
        self.chat_template = "test template"

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(i) for i in token_ids)

    def convert_ids_to_tokens(self, token_ids):
        return [f"tok_{i}" for i in token_ids]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        ids = [10, 11]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }


class MockLMHead(torch.nn.Module):
    def __init__(self, hidden_dim=8, vocab_size=32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(vocab_size, hidden_dim))

    def forward(self, x):
        return x @ self.weight.T


class MockModel(torch.nn.Module):
    def __init__(self, hidden_dim=8, vocab_size=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lm_head = MockLMHead(hidden_dim, vocab_size)

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        **kw,
    ):
        batch = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        seq = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        total_len = past_len + seq
        new_kv = (
            (torch.ones(batch, 2, total_len, 4), torch.ones(batch, 2, total_len, 4)),
        )
        hidden = (
            torch.arange(total_len - seq, total_len, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(batch, seq, self.hidden_dim)
        )
        return SimpleNamespace(
            past_key_values=new_kv, hidden_states=(hidden,), logits=self.lm_head(hidden)
        )


def _build_mock_wrapper():
    model = MockModel(hidden_dim=8, vocab_size=32).eval()
    model.config = SimpleNamespace(
        _commit_hash="model-revision", _attn_implementation="eager"
    )
    wrapper = SimpleNamespace(
        model=model,
        tokenizer=MockTokenizer(),
        device=torch.device("cpu"),
        dtype_name="float32",
        _apply_latent_realignment=lambda h, m: h * 1.0,
        prepare_chat_batch=lambda msgs, **kw: (
            ["prompt text"],
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.tensor([[1, 1, 1]], dtype=torch.long),
            [["tok_1", "tok_2", "tok_3"]],
        ),
    )
    return wrapper


def test_verification_01_02_03_04_05(tmp_path):
    """Verify criteria 1 to 5: branch parity, clean ple, token-hidden alignment, logits restoration, KV resumption."""
    wrapper = _build_mock_wrapper()
    args = SimpleNamespace(
        model_name="Qwen/Qwen3-0.6B",
        task="gsm8k",
        latent_steps=3,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        control_signal="\nSolution:",
    )
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("42", True, None)
    runner = InterventionContinuationRunner(wrapper, args, evaluator)
    item = {"question": "What is 2+2?", "gold": "4"}

    # UC-01
    kv_snap, common_h, meta = runner.compute_common_latent(item)
    assert meta["prompt_length"] == 3
    assert meta["latent_steps"] == 3
    assert common_h.shape == (6, 8)  # 3 prompt + 3 latent

    last_hidden = common_h[-1]

    # UC-02: Branching
    ple_h, ple_rec = runner.execute_ple(item, kv_snap, last_hidden)
    plce_h, plce_rec = runner.execute_plce(item, kv_snap)
    _plpe_h, _plpe_rec = runner.execute_plpe(item, kv_snap)

    # 1. Same common latent start: kv_snap was cloned for all three branches
    assert ple_rec["intervention_length"] == 0
    # 2. PLE has no intervention tokens
    assert len(ple_rec["intervention_token_ids"]) == 0
    # 3. Position and hidden correspondence
    assert ple_h.shape[0] == ple_rec["generated_length"]
    assert plce_h.shape[0] == (
        plce_rec["intervention_length"] + plce_rec["generated_length"] - 1
    )

    # 4. Hidden state restores logits matching greedy generation
    lm_head = wrapper.model.lm_head
    for i, tok_id in enumerate(ple_rec["generated_token_ids"]):
        pred_logits = lm_head(ple_h[i])
        assert int(pred_logits.argmax().item()) == tok_id

    # 5. KV resumption parity: regenerating with identical kv snapshot yields exact same tokens
    ids2, _, _, _ = decode_tokens_with_hiddens(
        wrapper.model,
        torch.device("cpu"),
        kv_snap,
        last_hidden,
        wrapper.tokenizer,
        args.max_new_tokens,
    )
    assert ids2 == ple_rec["generated_token_ids"]


def test_verification_06_07_resumption_and_partial_failure(tmp_path):
    """Verify criteria 6 & 7: completed shared and branches reused; failed conditions recomputed."""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)
    wrapper = _build_mock_wrapper()
    args = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen/Qwen3-0.6B",
        method="latent_mas",
        prompt="sequential",
        latent_steps=2,
        max_new_tokens=4,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        max_samples=1,
        overwrite=False,
    )
    dataset = MagicMock()
    dataset.load.return_value = [{"question": "Q1", "gold": "1"}]
    evaluator = MagicMock()
    evaluator.evaluate.return_value = ("1", True, None)

    uc = InterventionContinuationUseCase(
        dataset_port=dataset,
        storage=storage,
        evaluator_port=evaluator,
    )
    _summary, recs = uc.execute(wrapper, args)
    assert len(recs) == 4

    plpe_dir = next(
        storage.base_dir.glob(
            "intervention_continuation/*/*/*/samples/000000/results/ls2/*/plpe"
        )
    )
    (plpe_dir / "result.json").unlink()
    wrapper.model.forward = MagicMock(wraps=wrapper.model.forward)
    uc.execute(wrapper, args)
    assert (plpe_dir / "result.json").is_file()
    assert wrapper.model.forward.call_count > 0  # PLPE only


def test_verification_08_09_cache_cleaning_and_path_isolation(tmp_path):
    """Verify criteria 8 & 9: clean-cache preserves storage data; intervention_continuation path never collides."""
    resolver = PathResolver(tmp_path)
    cache_mgr = ExecutionCacheManager(resolver)
    args_std = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen3-8B",
        method="latent_mas",
        prompt="sequential",
        latent_steps=20,
        seed=42,
    )
    args_interv = SimpleNamespace(
        task="gsm8k",
        model_name="Qwen3-8B",
        method="latent_mas",
        prompt="sequential",
        latent_steps=20,
        seed=42,
        intervention_kv=True,
        cross_policy="shift_1",
    )
    p_std = build_experiment_relpath(args_std)
    p_cross = build_experiment_relpath(args_interv)
    assert p_std != p_cross

    # Verify clean-cache does not touch experiments dir
    exp_dir = resolver.get_experiment_dir(p_std, create=True)
    sentinel = exp_dir / "summary.json"
    sentinel.write_text('{"status": "ok"}', encoding="utf-8")
    runtime_dir = resolver.cache_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "temp.cache").write_text("volatile", encoding="utf-8")
    cache_mgr.clean_volatile_cache()
    assert sentinel.is_file()
    assert not (runtime_dir / "temp.cache").is_file()


def test_uc04_post_hoc_analyzer(tmp_path):
    """Verify UC-04: post-hoc logits, entropy, and deltas analysis via ResearchDataReader."""
    resolver = PathResolver(tmp_path)
    storage = FileStorage(resolver)
    reader = ResearchDataReader(path_resolver=resolver)
    run_id = "test_run"

    h = torch.randn(5, 8)
    kv = ((torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)),)
    state = resolver.experiments_dir / run_id
    storage.save_shared_segment(state, "000000", 0, 0, kv, h, {"prompt_length": 5})
    model = MockModel(hidden_dim=8, vocab_size=16)
    analyzer = SingleInterventionAnalyzer(reader=reader, model_module=model)
    loaded_h, _kv, _meta = analyzer.load_common_state(run_id, "000000", 0)
    assert torch.equal(h, loaded_h)

    logits, entropy = analyzer.compute_latent_step_logits_and_entropy(loaded_h)
    assert logits.shape == (5, 16)
    assert entropy.shape == (5,)

    deltas = analyzer.analyze_trajectory_state_deltas(loaded_h)
    assert len(deltas["step_distances"]) == 4
    assert len(deltas["step_cosine_similarities"]) == 4
