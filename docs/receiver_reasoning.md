# Receiver reasoning exploration

`ReceiverReasoningUseCase` runs one paired 2×2 experiment, independently of
receiver acquisition and intervention. `--upstream_steps 10,40` specifies low and
high compute within a single run; use a comma-separated string in YAML rather
than a sweep list. Latent steps apply to every non-judger agent.

For each sample and compute level, upstream inference runs once. Both receiver
modes receive independent deep copies of that same cache. Seed, temperature,
and top-p are identical across cells; the seed is reset before each inference.
`free` retains the existing judger prompt and generation behavior.
`answer_only` uses an explicit final-answer prompt and renders the chat
with `enable_thinking=False`, bypassing any manual `<think>` prefix. For boxed-answer tasks, the assistant
prompt prefills `\boxed{`; the completion is joined to this prefix before evaluation.
Generated-token counts exclude the prefilled prefix. Both use
the same task evaluator. This is a prompt/template manipulation, not a hard
guarantee that every model will obey; inspect raw outputs before interpreting
results. The 64-token answer-only limit permits answer serialization, while
free reasoning defaults to 2048 tokens. Token-limit counts expose truncation.

```bash
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2
```

Artifacts follow the existing path resolver under
`.cache/evaluation/runs/receiver_reasoning_*` and are uploaded to MLflow:

- `sample_results.jsonl`: sample key/index, question, gold, compute level,
  receiver mode, shared context ID/length, parsed prediction, raw output,
  correctness, evaluator error, generated tokens, token-limit status, model,
  seed, resolved configuration, and agent traces. Progress is persisted after
  each cell, retaining partial results if inference fails.
- `summary.json`: per-cell accuracy, mean/median generated tokens, complete
  prediction frequencies and unique counts, paired correctness transitions,
  `D(S) = Acc(S, free) - Acc(S, answer_only)`, and
  `substitution_signal = D(low) - D(high)`.
- `resolved_config.yaml`: execution configuration and Git revision.

The summary contains descriptive results only. A positive substitution signal
is exploratory and provides no causal or statistical claim. Prediction
concentration and raw answer-only behavior require inspection. No intervention,
permutation test, seed repetition, or receiver budget sweep is performed.

## Offline MLflow analysis

```bash
just analyze-handoff --version-tag pilot-v1
just analyze-handoff --tracking-uri http://localhost:5000 \
  --version-tag pilot-v1 --bootstrap-count 5000 --seed 0 \
  --output-dir artifacts/latent_handoff
```

Run selection requires FINISHED status and exact equality of the MLflow
`version_tag` tag.
`--version-tag` is required. The selected experiment identifies the data source;
Git commit, model, and experiment-type tags do not filter its runs. All matching
finished runs are included, including multiple runs of the same model. Running,
failed, and killed runs are excluded.
Missing artifacts cause an explicit error rather than silent exclusion.
`--run-id` and `--model-name` selection options have been removed.

For new inference runs, set the tag using the existing config/CLI:

```bash
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k \
  --model_name Qwen/Qwen3-4B --version_tag pilot-v1
```

Git revision remains provenance metadata. Untagged historical runs do not
match any requested version; the analyzer does not infer or assign a version.

MLflow is the source for run metadata, logged metrics, resolved configuration,
and sample records. Downloads use `PathResolver` under
`.cache/receiver_reasoning/mlflow/<run_id>`. Execution artifacts need not exist
locally, and model/dataset inference is never called. Tracking URI follows the
existing local SQLite / `MLFLOW_TRACKING_URI` policy unless explicitly supplied.

Each selected run is analyzed independently, including separate sample tables
and bootstrap statistics. Results are never pooled across models or runs.
Reports are stored under `<output>/<model>/<run_id>/`. The root `summary.md`
links to individual reports, with its metadata in `reports.json`.
Each report includes `summary.md`, `metrics.json`, `sample_matrix.csv`,
`prediction_distribution.csv`, `bootstrap_statistics.json`, and full source
records in `raw_outputs.jsonl`. Markdown reads exported JSON/CSV only.
CSV values use ordinary quoting, with empty fields for missing values.
Four selected raw-output excerpts are capped at 1200 characters each.

Integrity checks cover duplicate/missing samples, all four sample sets,
question/gold consistency, model/seed/generation settings, and paired context
IDs/lengths. Invalid paired comparisons have null estimates/CIs with explicit
reasons and appear as N/A. The analyzer never uses a sample-set intersection.
Valid comparisons use their full identical sample sets.

Bootstrap resamples whole sample rows and reuses draws for metrics with the
same sample sets. Default: 5000 replicates, seed 0, percentile 95% CI. All
replicate values and sample ordering are retained. D(low), D(high), both
receiver accuracy deltas, and substitution_signal include CIs and paired n.

Parse failures are missing/empty parsed predictions, shown as
`<PARSE_FAILURE>` in frequencies and excluded from unique prediction counts.
Truncation means reaching the configured token limit, including EOS at the
limit. Repeated-prediction subsets select values appearing at least twice in
a cell, excluding parse failures. No interpretation or conclusion is added.

Markdown uses the original zero-based `sample_index`, unique within a run.
Integrity checks verify one index per sample and one sample per index across
all four cells. Original sample hashes remain in CSV/JSON for joins.
