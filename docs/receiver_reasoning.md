# Receiver reasoning exploration

`ReceiverReasoningUseCase` runs a paired 3×2 experiment, independently of
receiver acquisition and intervention. `--upstream_steps 10,40` specifies low and
high compute within a single run, plus a zero-compute no-handoff baseline; use a comma-separated string in YAML rather
than a sweep list. Latent steps apply to every non-judger agent.

For each sample and compute level, upstream inference runs once. Both receiver
modes receive independent deep copies of the same upstream KV cache. The default
handoff transfers the full cache, including upstream prompt and latent positions.
Use `--handoff_positions 10` to retain the tail-slicing condition: all upstream
steps run first, then only the final 10 positions are handed off. This lets the
10/40-step tail comparison keep transfer width fixed. No-handoff runs only the
receiver prompts with no upstream inference or cache.

The use case fixes temperature=0 and top-p=1 (greedy decoding), even when CLI
overrides request sampling. Resolved MLflow configuration records effective
`handoff_mode` (`full` or `tail`) and `handoff_positions` (null for full). Seed is
reset before each inference. One or two positive increasing upstream levels are
supported. A single level such as `--upstream_steps 5` produces a 2×2 matrix with
its baseline, without between-level compute contrasts. For a tail handoff, the
source cache must contain at least the requested positions.
`free` retains the existing judger prompt and generation behavior.
`answer_only` uses an explicit final-answer prompt and renders the chat
with `enable_thinking=False`, bypassing any manual `<think>` prefix. For boxed-answer tasks, the assistant
prompt prefills `\boxed{`; the completion is joined to this prefix before evaluation.
Generated-token counts exclude the prefilled prefix. Both use
the same task evaluator. This is a prompt/template manipulation, not a hard
guarantee that every model will obey; inspect raw outputs before interpreting
results. The 64-token answer-only limit permits answer serialization, while
free reasoning defaults to 4096 tokens in the exploration preset. Token-limit counts expose truncation.

```bash
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k
just run-receiver-reasoning -c lmas/receiver_reasoning/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2
```

Artifacts follow the existing path resolver under
`.cache/evaluation/runs/receiver_reasoning_*` and are uploaded to MLflow:

- `sample_results.jsonl`: sample key/index, question, gold, compute level,
  receiver mode, handoff condition/width, original upstream cache length, actual receiver budget, shared context ID/length, parsed prediction, raw output,
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
One excerpt per cell are capped at 1200 characters each.

Integrity checks cover duplicate/missing samples, all cell sample sets,
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
all cells. Original sample hashes remain in CSV/JSON for joins.

New runs include `include_no_handoff` and `handoff_positions` in their resolved
configuration. The analyzer detects baseline cells automatically, exports all six
cells, and adds D(no-handoff) and paired accuracy gains over no-handoff for each
upstream level and receiver mode. Historical four-cell runs retain their original
comparisons and budgets; they are not assigned baseline or fixed-width results.

Context construction and receiver decoding remain separate domain operations,
coordinated by the existing application use case. Future matched/mismatched
preflight can reuse these operations; no preflight is implemented here.

## MLflow traces

Each sample has a `receiver_reasoning_sample` root trace linked to its run,
with one build span per upstream level and one receiver LLM span per cell. Build spans contain
upstream agent prompts and original cache length. Receiver spans contain the
actual prompt, output, token usage, shared context identity, handoff width,
budget, correctness and token-limit status. Ground truth and per-cell correctness
and truncation assessments are logged on the sample trace. Exceptions retain
failed spans, partial records and a FAILED run; pending traces flush before run
termination.

Load the configured remote tracking URI explicitly when running outside direnv:

```bash
uv run --env-file .env python -m src.run --receiver_reasoning \
  -c lmas/receiver_reasoning/gsm8k --model_name Qwen/Qwen3-0.6B \
  --max_samples 1 --max_new_tokens 256 \
  --tracking_experiment_name latent_handoff_pilot_refined_trace_smoke_20261001 \
  --version_tag pilot-trace-smoke
```

This short-budget smoke verifies execution and remote tracing, not performance
or the exploration preset's 4096-token free budget.

## Experimental measurements

The resolved configuration, experiment/run naming, version tags and existing
provenance fields retain the Pilot's conventions. Additional measurements are
experiment results, not a replacement environment metadata schema.

- `runtime`: model-load time; evaluation wall time and time per sample;
  synchronized upstream-build and receiver-decode times; receiver output and
  prompt token totals; unique upstream prompt tokens and latent steps; receiver
  tokens/sec; peak allocated and reserved GPU bytes.
- `execution`: expected/completed/successful cell counts, evaluator errors,
  parse failures, empty outputs and generation-limit counts.
- `upstream_statistics`: context counts and mean/median/min/max/total original
  and handed-off KV positions/bytes, layer counts, prompt tokens, agent counts,
  latent steps and build times by upstream level.
- Each cell: accuracy/counts, token min/max/mean/median, prediction distribution,
  parse/error/empty-output counts, truncation rate and inference/cache diagnostics.
- `paired_comparisons`: receiver-mode, upstream-level and no-handoff comparisons;
  accuracy differences, prediction-change rates, paired harm/rescue counts and
  joint correctness. Six-cell correctness patterns remain in the summary artifact.
- Sample records and spans carry prompt/response, prompt token counts, cache
  presence/length/bytes/layers/dtype, original cache size, executed latent steps,
  synchronized latency, parse/execution status and trace identity. Ground truth,
  optional reference solution and execution-success assessments join existing
  per-cell correctness/truncation assessments.

Evaluation wall time starts after model loading and includes sample tracking and
record persistence. Inference times synchronize the selected CUDA device and
exclude trace/artifact I/O; decode time includes prompt preparation, cache cloning
and evaluation. Receiver tokens/sec divides generated token IDs by receiver
inference time. A paired upstream context is built once: costs repeated on both
cell records are deduplicated by context ID for run totals and upstream statistics.
Latent steps sum actual non-judger agent traces, separately from generated text.
Token counts exclude prompt/cache positions and the answer-only prefill.

`peak_vram_bytes` is PyTorch's maximum allocated memory, matching the existing
benchmark definition; `peak_vram_reserved_bytes` reports allocator reservations.
Run aggregates include model loading: the entry point resets counters before
loading, and the Pilot retains the maximum before each subsequent scope reset.
CPU runs report zero. These are not system-wide NVML measurements
or estimates of CPU RAM; cache byte counts measure tensor payloads.

Failures persist `results/failure.json` with the full traceback, completed-record
counts and measured runtime/memory, alongside resolved config and partial records.
The run and failed spans are marked accordingly. Numeric diagnostics are logged
to MLflow; distributions, joint correctness patterns and full outputs stay in
artifacts. Offline analysis exports per-cell diagnostics and sample measurements,
leaving unrecorded historical measurements unavailable.

## Trace and span measurement scopes

Each sample root trace records its own peak allocated/reserved VRAM and wall time,
plus completed cell counts, execution health, output/prompt token totals, actual
latent steps, upstream/receiver inference totals and receiver tokens/sec. Shared
upstream work is counted once. Failed samples retain completed-cell totals and
memory measurements in their root span.

Every upstream build and receiver decode span also records peak allocated and
reserved VRAM as numeric attributes. These are measured independently: synchronize
CUDA, snapshot starting allocated/reserved bytes, reset peak counters, execute,
synchronize and capture the peak. Nested measurements retain parent maxima before
each reset, so earlier samples do not contaminate later sample or decode peaks.
Exceptions still publish measurements. Receiver spans include tokens/sec.

`vram_start_allocated_bytes` and `vram_start_reserved_bytes` expose each scope's
baseline. `incremental_peak_vram_bytes` and
`incremental_peak_vram_reserved_bytes` are the scope peak minus that baseline.
Absolute peaks include resident model weights and any live shared KV. Reserved
memory may carry over from the allocator's pool; no empty-cache operation is
inserted to change the execution being measured. Run peaks remain the maximum
across model loading and all samples, rather than the final decode's reset counter.

Sample records carry `sample_*`, `upstream_*` and `receiver_*` scoped memory
measurements. Summary/MLflow cell diagnostics and offline reports aggregate these
recorded peaks. Existing run-only results remain run-only; no historical sample
or span peak is inferred. Model-load time and population-level accuracy,
distributions and paired comparisons remain run aggregates.

The trace table exposes sample costs through trace tags: `peak_vram_gib`,
`peak_reserved_vram_gib`, `output_tokens_total`, and `latent_steps_total`. Values use GiB (2³⁰ bytes). Select
these under **Columns → Tags**; span attributes remain available in the trace
details. These tags contain measured costs, while existing provenance tags stay
unchanged. Run metrics include core cell comparisons, run costs, execution health,
and paired accuracy deltas. Detailed distributions remain in JSON artifacts and
traces rather than being expanded into hundreds of run metrics.
