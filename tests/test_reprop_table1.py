"""Table 1 preset expansion, dtype selection, tokens, and result identity."""

import argparse

import pytest
import torch
from torch import nn

from latentmas_reprop.infrastructure.cache.manager import ExecutionCacheManager
from latentmas_reprop.infrastructure.models.dtype import (
    ensure_cuda_available,
    resolve_model_dtype,
)
from latentmas_reprop.infrastructure.models.realignment import RealignmentManager
from latentmas_reprop.infrastructure.paths.resolver import PathResolver
from run.cli import parse_run_matrix


def _args(**overrides):
    values = {
        "method": "latent_mas",
        "model_name": "Qwen/Qwen3-4B",
        "task": "gsm8k",
        "prompt": "sequential",
        "latent_steps": 20,
        "seed": 42,
        "max_new_tokens": 2048,
        "temperature": 0.6,
        "top_p": 0.95,
        "generate_bs": 1,
        "max_samples": -1,
        "think": False,
        "latent_space_realign": False,
        "use_vllm": False,
        "split": "test",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_table1_presets_expand_without_mixing_diagnostic_or_14b():
    latent_4b = parse_run_matrix(["-c", "lmas/reprop/lm_q34_gsm8k", "--dry-run"])
    latent_8b = parse_run_matrix(["-c", "lmas/reprop/lm_q38_gsm8k", "--dry-run"])
    latent_14b = parse_run_matrix(["-c", "lmas/reprop/lm_q314_gsm8k", "--dry-run"])
    baselines_4b = parse_run_matrix(["-c", "lmas/reprop/bs_q34_gsm8k", "--dry-run"])
    baselines_8b = parse_run_matrix(["-c", "lmas/reprop/bs_q38_gsm8k", "--dry-run"])
    baselines_14b = parse_run_matrix(["-c", "lmas/reprop/bs_q314_gsm8k", "--dry-run"])
    text_mas_4b = parse_run_matrix(["-c", "lmas/reprop/tm_q34_gsm8k", "--dry-run"])
    text_mas_8b = parse_run_matrix(["-c", "lmas/reprop/tm_q38_gsm8k", "--dry-run"])
    text_mas_14b = parse_run_matrix(["-c", "lmas/reprop/tm_q314_gsm8k", "--dry-run"])
    latent = [*latent_4b, *latent_8b, *latent_14b]
    baselines = [*baselines_4b, *baselines_8b, *baselines_14b]
    text_mas = [*text_mas_4b, *text_mas_8b, *text_mas_14b]

    assert [(run.model_name, run.latent_steps) for run in latent] == [
        ("Qwen/Qwen3-4B", 10),
        ("Qwen/Qwen3-4B", 20),
        ("Qwen/Qwen3-4B", 40),
        ("Qwen/Qwen3-8B", 10),
        ("Qwen/Qwen3-8B", 20),
        ("Qwen/Qwen3-8B", 40),
        ("Qwen/Qwen3-14B", 10),
        ("Qwen/Qwen3-14B", 20),
        ("Qwen/Qwen3-14B", 40),
    ]
    assert [(run.method, run.model_name) for run in baselines] == [
        ("baseline", "Qwen/Qwen3-4B"),
        ("baseline", "Qwen/Qwen3-8B"),
        ("baseline", "Qwen/Qwen3-14B"),
    ]
    assert [(run.method, run.model_name) for run in text_mas] == [
        ("text_mas", "Qwen/Qwen3-4B"),
        ("text_mas", "Qwen/Qwen3-8B"),
        ("text_mas", "Qwen/Qwen3-14B"),
    ]
    for run in [*latent, *baselines, *text_mas]:
        assert run.max_samples == -1
        assert run.generate_bs in {1, 2, 4, 8, 16, 32}
        assert run.max_new_tokens == 2048
        assert run.temperature == 0.6
        assert run.top_p == 0.95
        assert run.seed == 42
        assert run.use_vllm is False
        assert run.latent_space_realign is False
        assert run.think is False
        assert run.task == "gsm8k"
        assert run.split == "test"
        assert run.prompt == "sequential"


def test_cpu_dtype_is_fp16_and_unavailable_cuda_fails(monkeypatch):
    assert resolve_model_dtype(torch.device("cpu")) is torch.float16

    class _Cuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda",
        _Cuda,
    )
    with pytest.raises(RuntimeError, match="no available CUDA device"):
        ensure_cuda_available(torch.device("cuda"))


def test_cuda_dtype_is_fp16_regardless_of_bf16_support(monkeypatch):
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.ensure_cuda_available",
        lambda device: None,
    )
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda.current_device",
        lambda: 0,
    )
    # sm_70 (e.g. V100): even if PyTorch reports is_bf16_supported=True, fallback to float16
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda.get_device_capability",
        lambda idx: (7, 0),
    )
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda.is_bf16_supported",
        lambda: True,
    )
    assert resolve_model_dtype(torch.device("cuda")) is torch.float16

    # sm_80 without bf16 support -> float16
    monkeypatch.setattr(
        "latentmas_reprop.infrastructure.models.dtype.torch.cuda.is_bf16_supported",
        lambda: False,
    )
    assert resolve_model_dtype(torch.device("cuda")) is torch.float16


def test_available_cuda_does_not_require_exact_compiled_arch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_86", "sm_90"])
    ensure_cuda_available(torch.device("cuda"))


class _TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def _legacy_identity_normalize(hidden, model):
    """Previous path: solve the ridge system, replace it with I, then normalize."""
    input_weight = model.get_input_embeddings().weight.detach().float()
    output_weight = model.get_output_embeddings().weight.detach().float()
    gram = output_weight.T @ output_weight
    gram = gram + 1e-5 * torch.eye(gram.shape[0])
    solved = torch.linalg.solve(gram, input_weight.T @ input_weight)
    del solved
    matrix = torch.eye(input_weight.shape[1], dtype=torch.float32)
    target_norm = input_weight.norm(dim=1).mean()
    aligned = hidden.float() @ matrix
    aligned = aligned * (
        target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    )
    return aligned.to(hidden.dtype)


def test_disabled_realignment_matches_normalized_identity(tmp_path):
    torch.manual_seed(0)
    model = _TinyLM()
    hidden = torch.randn(2, 8)
    args = _args(latent_space_realign=False)
    manager = RealignmentManager(
        "tiny",
        ExecutionCacheManager(PathResolver(tmp_path)),
        args,
    )
    updated = manager.apply(hidden, model)
    reference = _legacy_identity_normalize(hidden, model)
    assert torch.allclose(updated, reference, rtol=1e-5, atol=1e-5)
    target = model.embed.weight.detach().float().norm(dim=1).mean()
    assert torch.allclose(
        updated.float().norm(dim=-1), target.expand(2), rtol=1e-5, atol=1e-5
    )


def test_disabled_realignment_does_not_solve_or_read_output(tmp_path, monkeypatch):
    class _InputOnly(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(4, 4)

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            raise AssertionError("output embeddings should not be read")

    monkeypatch.setattr(
        torch.linalg,
        "solve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("solve")),
    )
    manager = RealignmentManager(
        "tiny",
        ExecutionCacheManager(PathResolver(tmp_path)),
        _args(latent_space_realign=False),
    )
    hidden = torch.ones(1, 4)
    output = manager.apply(hidden, _InputOnly())
    assert torch.isfinite(output).all()
