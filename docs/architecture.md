# Architecture & System Design

This repository is structured around **Hexagonal Architecture (Ports & Adapters)**, strictly isolating core domain logic, application use cases, external infrastructure concerns, storage policies, and the command-line execution entry points.

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
│    - InterventionKvUseCase & InterventionContinuationUseCase           │
│    - RepresentationAcquisitionUseCase & RepresentationSenderProbeUseCase│
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
               ▼                                          ▼
┌──────────────────────────────────────┐   ┌─────────────────────────────┐
│ Domain Layer (latentmas_reprop/domain)│   │ Infrastructure (latentmas_  │
│  - Models: Agent, BenchmarkMetrics,  │   │ reprop/infrastructure)      │
│            TensorEntry, Manifest ... │◄──┤  - Adapters implementing    │
│  - Ports: ModelPort, DatasetPort,    │   │    Domain Ports             │
│           EvaluatorPort, CachePort,  │   │  - FileStorage              │
│           TrackingPort               │   │  - ResearchDataReader       │
│  - Services: LatentMAS, TextMAS,     │   │  - MLflowTracker            │
│              Baseline, Prompts       │   │  - ExecutionCacheManager    │
└──────────────────────────────────────┘   │  - ModelWrapper (HF / vLLM) │
                                           │  - Dataset Loaders          │
                                           │  - PathResolver             │
                                           └─────────────────────────────┘
```

### 1.1 Domain Layer (`src/latentmas_reprop/domain/`)
The core domain is free of external framework dependencies (such as CLI parsers, torch disk I/O, or remote servers):
- **Models (`domain/models/`)**: Data contracts including `Agent`, `ProblemSample`, `AgentTrace`, `EvaluationResult`, `BenchmarkMetrics`, `InterventionMetrics`, and storage exceptions.
- **Ports (`domain/ports/`)**:
  - `ModelPort`: Interface for model tokenization, text generation, and latent state retrieval.
  - `DatasetPort`: Interface for loading standardized problem samples.
  - `EvaluatorPort`: Interface for answer extraction and ground-truth verification.
  - `CachePort`: Interface for execution cache management and safe cleanup.
  - `ExperimentTrackerPort`: Interface for metrics, parameters, and GenAI tracing spans.
- **Services (`domain/services/`)**:
  - `LatentMASMethod`: Multi-agent communication via latent KV-cache representations.
  - `TextMASMethod`: Multi-agent collaboration via sequential/hierarchical text reasoning traces.
  - `BaselineMethod`: Single-agent baseline execution.
  - `prompts/`: Prompt builder and agent topology definitions.

### 1.2 Application Layer (`src/latentmas_reprop/application/`)
Orchestrates domain services and ports to fulfill use-case workflows (see [**Application Use Cases Catalog**](use_cases/README.md)):
- [**`BenchmarkUseCase`**](use_cases/benchmark.md): Orchestrates loading problem batches, invoking multi-agent reasoning, evaluating answers, recording agent traces, and persisting execution run caches.
- [**`InterventionKvUseCase`**](use_cases/intervention_kv.md): Manages paired condition interventions (`own`, `cross`, `drop`, `zero`) with per-condition file storage and crash-safe resumption.
- [**`InterventionContinuationUseCase`**](use_cases/intervention_continuation.md): Manages common latent calculation and independent branching (`ple`, `plce`, `plpe`) with 10-step incremental checkpointing.
- [**`RepresentationAcquisitionUseCase`**](use_cases/representation_acquisition.md): Evaluates carrier digit recovery across latent, prompt, and no-cache contexts with per-condition file storage.
- [**`RepresentationSenderProbeUseCase`**](use_cases/representation_sender_probe.md): Analyzes saved sender latent states with logistic regression and max-statistic permutation testing (GPU/CPU).

### 1.3 Infrastructure Layer (`src/latentmas_reprop/infrastructure/`)
Contains technology-specific adapters and system integration:
- **`paths/resolver.py`**: `PathResolver` manages all physical filesystem locations (`root`, `storage`, `experiments`, `mlflow`, `artifacts`, `cache`, `data`, `configs`).
- **`storage/tensor_store.py`**: `FileStorage` provides condition-based directory management, atomic file saving, and condition completion verification.
- **`storage/reader.py`**: `ResearchDataReader` provides a unified read interface for downstream analysis scripts across MLflow and file-based experiment directories.
- **`tracking/mlflow_tracker.py`**: `MLflowTracker` implements `ExperimentTrackerPort`, directing persistent tracking to `storage/mlflow/mlflow.db` and linking runs to local experiment data directories.
- **`cache/manager.py`**: `ExecutionCacheManager` implements `CachePort`, writing volatile caches under `.cache/` and supporting `clean_volatile_cache()`.
- **`models/model_wrapper.py`**: `ModelWrapper` implements `ModelPort`, handling HuggingFace transformers (`AutoModelForCausalLM`) and vLLM backends.
- **`datasets/`**: Loaders for GSM8K, AIME, ARC, MedQA, MBPP+, HumanEval+, etc.
- **`evaluators/`**: Answer extraction and verification adapters.

### 1.4 Execution Layer (`src/run/`)
The executable command-line interface:
- **`src/run/cli.py`**: CLI argument parsing supporting configuration YAML files via `-c`, with explicit flag overrides.
- **`src/run/runner.py`**: Environment bootstrap, device selection, port wiring, and execution.
- **`src/run/__main__.py`**: Python module entry point (`python -m src.run`).
- **`justfile`**: Task automation recipes (`just run`, `just test`, `just lint`, `just format`, `just clean-cache`, `just mlflow-ui`).
