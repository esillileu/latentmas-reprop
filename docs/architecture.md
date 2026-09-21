# Architecture & System Design

This repository is structured around **Hexagonal Architecture (Ports & Adapters)**, strictly isolating core domain logic, application use cases, external infrastructure concerns, and the command-line execution entry point.

---

## 1. Architectural Layers

```
                       ┌─────────────────────────┐
                       │   Execution Layer       │
                       │   (src/run, justfile)   │
                       └────────────┬────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  Application Layer (src/latentmas_reprop/application)                  │
│    - BenchmarkUseCase (coordinates execution, caching, evaluation)     │
│    - EvaluationService (computes accuracy and metrics)                │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
               ▼                                          ▼
┌──────────────────────────────────────┐   ┌─────────────────────────────┐
│ Domain Layer (latentmas_reprop/domain)│   │ Infrastructure (latentmas_  │
│  - Models: Agent, ProblemSample, ... │   │ reprop/infrastructure)      │
│  - Ports: ModelPort, DatasetPort,    │◄──┤  - Adapters implementing    │
│           EvaluatorPort, CachePort   │   │    Domain Ports             │
│  - Services: LatentMAS, TextMAS,     │   │  - CacheManager             │
│              Baseline, Prompts       │   │  - ModelWrapper (HF / vLLM) │
└──────────────────────────────────────┘   │  - Dataset Loaders          │
                                           │  - PathResolver             │
                                           └─────────────────────────────┘
```

### 1.1 Domain Layer (`src/latentmas_reprop/domain/`)
The core domain is free of external framework dependencies (such as CLI parsers or disk I/O details):
- **Models (`domain/models.py`)**: Data contracts including `Agent`, `ProblemSample`, `AgentTrace`, `EvaluationResult`, `BenchmarkMetrics`, and `MethodType`.
- **Ports (`domain/ports/`)**:
  - `ModelPort`: Interface for model tokenization, text generation, and latent state retrieval.
  - `DatasetPort`: Interface for loading standardized problem samples.
  - `EvaluatorPort`: Interface for answer extraction and ground-truth verification.
  - `CachePort`: Interface for persistent execution caching and artifact storage.
- **Services (`domain/services/`)**:
  - `LatentMASMethod`: Multi-agent communication via latent KV-cache representations.
  - `TextMASMethod`: Multi-agent collaboration via sequential/hierarchical text reasoning traces.
  - `BaselineMethod`: Single-agent baseline execution.
  - `prompts.py`: Prompt builder and agent topology definitions.

### 1.2 Application Layer (`src/latentmas_reprop/application/`)
Orchestrates domain services and ports to fulfill use-case workflows:
- **`BenchmarkUseCase`**: Orchestrates loading problem batches, invoking multi-agent reasoning, evaluating answers, recording agent traces, and persisting execution run caches.
- **`EvaluationService`**: Aggregates batch scores, calculating overall accuracy and elapsed runtimes.

### 1.3 Infrastructure Layer (`src/latentmas_reprop/infrastructure/`)
Contains technology-specific adapters and system integration:
- **`paths/resolver.py`**: `PathResolver` manages all physical filesystem locations (configs, cache, data files).
- **`cache/manager.py`**: `ExecutionCacheManager` implements `CachePort`, writing runs and precomputed realignment matrices under `.cache/`.
- **`models/model_wrapper.py`**: `ModelWrapper` implements `ModelPort`, handling HuggingFace transformers (`AutoModelForCausalLM`) and vLLM backends.
- **`datasets/`**: Loaders for GSM8K, AIME, ARC, MedQA, MBPP+, HumanEval+, etc.
- **`evaluators/`**: Answer extraction and verification adapters.

### 1.4 Execution Layer (`src/run/`)
The executable command-line interface:
- **`src/run/cli.py`**: CLI argument parsing supporting configuration YAML files via `-c`, with explicit flag overrides.
- **`src/run/runner.py`**: Environment bootstrap, device selection, port wiring, and execution.
- **`src/run/__main__.py`**: Python module entry point (`python -m src.run`).
- **`justfile`**: Task automation recipes (`just run`, `just test`, `just lint`, `just format`, `just clean-cache`).
