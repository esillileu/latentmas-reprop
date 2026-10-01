# LatentMAS Reprop

A refactored, production-ready reproduction and benchmarking framework for **LatentMAS (Latent Collaboration in Multi-Agent Systems)** built with Hexagonal Architecture, unified path/cache management, and `just` task runner integration.

---

## 📖 Documentation Index

Use the links below to navigate the project documentation:

| Document | Description |
|---|---|
| [**Architecture & Design**](docs/architecture.md) | Hexagonal Architecture (Ports & Adapters), domain models, ports, and layer separation |
| [**Usage & Execution**](docs/usage.md) | Running benchmarks with `just run`, CLI options, and recipe reference |
| [**Configuration Guide**](docs/configuration.md) | YAML preset schemas (`configs/`), naming conventions (`{method}_{model}_{task}.yaml`), and customization |
| [**Cache & Path Management**](docs/cache_and_paths.md) | Unified `.cache/` hierarchy and central `PathResolver` system |
| [**Development Guide**](docs/development.md) | Developer setup, tests (`just test`), linter (`just lint`), and extending adapters |

---

## 🚀 Quick Start

### 1. Installation
```bash
# Clone and install dependencies
git clone https://github.com/esillileu/latentmas-reprop.git
cd latentmas-reprop
uv sync
```

### 2. Run Benchmarks
Run experiments directly with `just run -c <preset>`:

```bash
# Run LatentMAS preset on GSM8K (Qwen3-8B)
just run -c lmas/reprop/lm_q38_gsm8k

# Run with option overrides
just run -c lmas/reprop/lm_q38_gsm8k --max_samples 10 --generate_bs 2

# Run Single Baseline, TextMAS, or LatentMAS presets (Qwen3-8B)
just run -c lmas/reprop/bs_q38_gsm8k
just run -c lmas/reprop/tm_q38_gsm8k
just run -c lmas/reprop/lm_q38_gsm8k
```

Run the secret-digit receiver acquisition experiment (transformers backend only):

```bash
just run-acquisition -c lmas/secret_digit/preflight \
  --model_name Qwen/Qwen3-0.6B --latent_steps 4
```

This decomposes full, prompt-only, latent-only, and no-cache digit transfer without
generating text. Sender states are collected with frozen-model inference and analyzed
with grouped five-fold logistic regression plus max-statistic permutation correction.
GPU analysis batches independent permutation classifiers with PyTorch; no model
inference is needed when reanalyzing saved states.

Saved states can be analyzed without loading a model:

```bash
just analyze-sender-probe \
  --states .cache/evaluation/receiver_acquisition/sender_latent_states.pt \
  --source-config configs/lmas/secret_digit/preflight.yaml
```

Replace a legacy MLflow acquisition run while preserving its acquisition metrics,
artifacts, timestamps, and traces:

```bash
just analyze-sender-probe \
  --source-run-id RUN_ID \
  --replace-source-run \
  --backend torch \
  --permutations 5000 \
  --batch-size 256
```

The replacement is created and verified before the legacy run is soft-deleted.

### 3. Developer Commands
```bash
just test        # Run test suite
just lint        # Check linting and formatting
just format      # Auto-format code
just clean-cache # Clean local execution cache
```

---

## 🔗 Reference & Upstream

- **Original Paper**: [Latent Collaboration in Multi-Agent Systems](https://arxiv.org/abs/2511.20639) (ICML 2026 Spotlight)
- **Original Repository**: [Gen-Verse/LatentMAS](https://github.com/Gen-Verse/LatentMAS)

```bibtex
@inproceedings{zou2025latentmas,
  title={Latent Collaboration in Multi-Agent Systems},
  author={Jiaru Zou and Ruizhong Qiu and Gaotang Li and Xiyuan Yang and Katherine Tieu and Pan Lu and Ke Shen and Hanghang Tong and Yejin Choi and Jingrui He and James Zou and Mengdi Wang and Ling Yang},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
See the [`LICENSE`](LICENSE) file for details.

Explore paired receiver answer-only versus free reasoning with shared upstream caches:

```bash
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k
# Small-model execution check
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2
```

Model-specific 20-sample presets are available as
`lmas/receiver_reasoning/lm_q34_gsm8k` (0/10/20 steps) and
`lmas/receiver_reasoning/lm_q30.6_gsm8k` (0/5/10 steps). Each keeps all conditions
in one paired run; use `--dry-run` to inspect the plan without inference.

The preset runs 20 GSM8K samples with Qwen3-4B at 10 and 40 latent steps,
passing the full upstream KV cache, plus no-handoff baselines. Both receiver
modes use greedy decoding; free reasoning has a 4096-token budget.
See [receiver reasoning exploration](docs/receiver_reasoning.md) for artifacts and controls.

Analyze saved Latent Handoff Pilot results from MLflow without inference:

```bash
just analyze-handoff --version-tag pilot-v1
```

The report and machine-readable paired/bootstrap results are written to
`artifacts/latent_handoff/<model>/<run_id>/` by default, with a report index
at `artifacts/latent_handoff/summary.md`. Finished runs matching the exact MLflow `version_tag` are analyzed separately
for every model, without pooling results or filtering by model/Git commit.
