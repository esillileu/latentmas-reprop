# Receiver compute preflight

The independent `ReceiverComputePreflightUseCase` measures GSM8K receiver accuracy
across upstream latent steps U, full-cache matched/mismatched/no-handoff conditions
M, and receiver token budgets R. The receiver reasoning pilot remains separate.

```bash
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k --dry-run
# Full experiment command; do not use for an implementation smoke.
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k
# Two-sample small-model smoke, including actual capped-generation comparisons.
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2 --upstream_steps 1,2 \
  --receiver_budgets 8,16,free --max_new_tokens 32 --bootstrap_count 100 --verify_prefix
```

Defaults: Qwen3-4B, U=10/20, R=64/128/256/512/1024/free, free cap=4096,
20 samples, greedy decoding (`temperature=0`, `top_p=1`). Override
`--upstream_steps`, `--receiver_budgets`, `--max_new_tokens`,
`--target_accuracies`, `--bootstrap_count`, and `--seed`. Axes in YAML are
comma-separated strings; YAML lists expand independent runs in the existing CLI.
At least two unique samples are required for mismatched handoff.

Each sample/U upstream context is built exactly once. Full caches are retained in
CPU memory and cloned back to the model device for generation. No KV positions
are removed. Donors are a bijective deterministic rotation of samples sorted by
(cache length, sample index), separately for every U. Every donor differs from
its recipient; recipient/donor IDs, lengths, cache measurements and upstream
agent inputs are saved. This length-neighbor policy is explicit: the existing
intervention algorithm does not implement a length-matched policy.

Every receiver condition uses the same free prompt and decoding regime. One free
trajectory is generated per condition, and each finite R evaluates the actual
generated IDs[:R], including generated EOS tokens. Text is decoded directly from
those IDs and is never re-tokenized. No-handoff trajectories are generated once
per sample and reused for every U. `--verify_prefix` checks actual capped token
IDs, prompts, predictions and correctness at every finite R; it requires exactly
two samples to keep verification within smoke scope.

A free trajectory reaching its cap is flagged and its free cell is invalid.
Finite prefixes remain evaluable. No invalid sample is dropped to change the
paired sample set: any cell containing an invalid record has N/A accuracy/CI,
and comparisons involving that cell have N/A differences/CI. Exact common sample
sets, one row per cell, and the configured full Cartesian grid are required;
violations raise an error before paired statistics are computed.

All accuracy and difference CIs use percentile paired bootstrap resampling of
sample IDs, with the same resample indices for all U/M/R. Comparisons include
matched minus mismatched (pairing gain) and each handoff minus no-handoff at each
budget. R* is the smallest **tested finite** budget whose point accuracy reaches
the configured target; otherwise N/A. No interpolation, monotonic smoothing,
interpretation, or automated conclusion is generated. The free endpoint is
reported separately, since it is an uncapped reasoning regime rather than a
fixed finite budget. Bootstrap resample indices and all distributions are saved.

MLflow is the source of truth. Execution writes artifacts to a temporary staging
directory, uploads them under `results/`, and removes staging afterward. Saved
artifacts include `sample_results.jsonl`, `resolved_config.yaml`, `summary.md`,
`metrics.json`, `sample_matrix.csv`, `budget_curves.csv`, and
`bootstrap_statistics.json`. Raw rows contain predictions, correctness, prefix
IDs/text/counts, free counts/cap flags, donor/context metadata, receiver prompts
and token IDs, model/seed and upstream measurements. Failures upload partial
records/config and `failure.json` and mark the run FAILED.

```bash
just analyze-receiver-compute-preflight --run-id RUN_ID \
  --target-accuracies 0.5,0.7,0.9 --bootstrap-count 2000
```

Reanalysis requires a FINISHED preflight run and downloads raw records/config
from MLflow afresh into a temporary directory. It needs no model or existing
local cache. Outputs default to `artifacts/receiver_compute_preflight/RUN_ID/`
and are uploaded to the source MLflow run under `analysis/`. `MLFLOW_TRACKING_URI`
uses the repository's usual tracker configuration. `source.json` records run
provenance. Run tests with `just lint && just test`.
