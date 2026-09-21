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
# Run LatentMAS preset on GSM8K
just run -c lm_q30.6_gsm8k

# Run with option overrides
just run -c lm_q30.6_gsm8k --max_samples 10 --generate_bs 2

# Run TextMAS or Baseline
just run -c tm_q30.6_gsm8k
just run -c bs_q30.6_gsm8k
```

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
