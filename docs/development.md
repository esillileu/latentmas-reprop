# Development & Extension Guide

This guide covers developer workflows, testing, code quality checks, and how to extend the framework with new adapters and domain services.

---

## 1. Development Setup

The project uses [`uv`](https://docs.astral.sh/uv/) for Python package management:

```bash
# Clone the repository and install dependencies
git clone <repo_url>
cd latentmas-reprop
uv sync
```

---

## 2. Quality Checks & Testing

The repository relies on `just` to automate developer tasks:

```bash
# Run all unit tests
just test

# Check linting and formatting
just lint

# Auto-format codebase
just format
```

All unit tests reside in [`tests/`](../tests/):
- `tests/test_paths.py`: PathResolver invariants and directory resolution.
- `tests/test_cache.py`: CacheManager operations and structure.
- `tests/test_cli.py`: CLI arguments and configuration overrides.
- `tests/test_evaluators.py`: Regex extraction routines across benchmarks.

---

## 3. Extending the Framework

Thanks to the Hexagonal Architecture, extensions are made by implementing ports in the infrastructure layer without modifying core domain logic.

### 3.1 Adding a New Dataset Adapter
1. Define a loader function in [`src/latentmas_reprop/infrastructure/datasets/loaders.py`](file:///home/esillileu/projects/latentmas-reprop/src/latentmas_reprop/infrastructure/datasets/loaders.py) returning a list of `ProblemSample`:
   ```python
   def load_my_dataset(split: str = "test") -> list[ProblemSample]:
       ...
   ```
2. Register the loader in [`src/latentmas_reprop/infrastructure/datasets/registry.py`](file:///home/esillileu/projects/latentmas-reprop/src/latentmas_reprop/infrastructure/datasets/registry.py):
   ```python
   DATASET_REGISTRY["my_task"] = load_my_dataset
   ```

### 3.2 Adding a New Evaluator Adapter
Implement an evaluator adhering to [`EvaluatorPort`](file:///home/esillileu/projects/latentmas-reprop/src/latentmas_reprop/domain/ports/evaluator_port.py):
```python
class MyCustomEvaluator(EvaluatorPort):
    def extract_answer(self, text: str) -> str | None:
        ...

    def evaluate(self, prediction: str | None, ground_truth: str) -> bool:
        ...
```

### 3.3 Adding a New Method
Create a service class implementing the reasoning logic under `src/latentmas_reprop/domain/services/`, coordinating agents and interacting with `ModelPort`.
