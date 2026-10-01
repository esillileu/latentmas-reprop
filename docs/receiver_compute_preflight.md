# Receiver compute preflight

`ReceiverComputePreflightUseCase` measures GSM8K receiver accuracy and cost across
upstream latent steps U, full-cache matched/mismatched/no-handoff conditions M,
and receiver token budgets R. The receiver reasoning pilot remains separate.

```bash
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k --dry-run
# Full experiment command; do not use for an implementation smoke.
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k
# Two-sample small-model smoke: exact capped-prefix equivalence and bounded retries.
just run-receiver-compute-preflight -c lmas/receiver_compute_preflight/gsm8k \
  --model_name Qwen/Qwen3-0.6B --max_samples 2 --upstream_steps 1,2 \
  --receiver_budgets 8,16,free --max_new_tokens 32 --free_max_new_tokens 64 \
  --bootstrap_count 100 --verify_prefix
```

The preset expands Qwen3-4B/8B/14B into separate sequential MLflow runs. Each uses
U=10/20, R=64/128/256/512/1024/free, 20 identical paired samples, and greedy decoding
(`temperature=0`, `top_p=1`). The initial free cap is 4096 and the retry ceiling
is 8192. Override `--model_name`, `--upstream_steps`, `--receiver_budgets`,
`--max_new_tokens`, `--free_max_new_tokens`, `--target_accuracies`,
`--bootstrap_count`, and `--seed`. U/R/target axes in YAML are comma-separated
strings; top-level YAML lists expand independent runs. At least two unique
samples are required for mismatched handoff.

## Final-answer scoring

All endpoints, including free, use strict explicit final-answer scoring. A
complete numeric `\boxed{...}` or an explicit `Final answer: ...` / `Final answer
is ...` is required. A plain intermediate number never counts. A final-answer
label's number needs a closing dollar delimiter, sentence terminator, newline,
or natural trajectory termination; a bare number at a cutoff is incomplete
because later tokens could add digits. A newer unfinished answer marker also
invalidates an earlier answer. Explicit thinking content is excluded, including
an open `<think>` in the receiver prompt. Missing/incomplete final answers record
`no_answer=true`, `prediction=null`, `correct=false`, and `error_msg=no_answer`.
The original GSM8K evaluator and pilot/benchmark behavior are unchanged.

Every receiver condition uses the same free prompt and decoding regime. Each
finite R evaluates actual generated IDs[:R], including generated EOS tokens.
Text is decoded directly from IDs, preserving final-answer delimiters, and is
never re-tokenized. `--verify_prefix` checks actual capped IDs, prompt metadata,
strict predictions and correctness at every finite R. It requires exactly two
samples. Timing differences between reruns are not treated as decoding failures.

## Free endpoint

Only cap-limited trajectories retry, doubling the cap up to
`--free_max_new_tokens`. Each retry uses the same prompt, seed, decoding and
unchanged upstream cache; its prefix must exactly match the previous attempt.
EOS at exactly the cap is natural termination and does not retry. Natural
trajectories are not rerun. No-handoff trajectories are generated once per
sample and reused across U, including their retry history and measurements.

An unresolved trajectory is explicitly `pathological=true` with an invalid
free row, saved attempt caps/counts/termination reasons/latencies, and a separate
`pathological_cases` report table. Finite prefixes remain evaluable. Free accuracy
and primary cost/CI are N/A for any condition containing an invalid row, as are
paired comparisons involving it. No sample is dropped or quietly counted as an
ordinary free failure. Thus a reported valid free endpoint has zero truncation;
a remaining pathological case prevents reporting that endpoint as baseline
performance on the full paired set. `free_initial_cap_reached`, `free_final_cap`,
`free_retry_count`, `free_naturally_terminated` and `free_cap_reached` distinguish
initial retries from unresolved final truncation. Natural termination alone does
not imply that a final answer is present or correct.

## Receiver cost

Accuracy, generated-token counts, synchronized receiver latency and deterministic
work proxies are primary curve metrics. A non-stopping generation observer records
elapsed wall time at each requested token budget, with CUDA synchronization at
measurement boundaries. Finite prefixes use their measured checkpoints; a
trajectory ending before R uses its full measured duration. No per-token speed
interpolation is used. Latency includes receiver prompt prefill and autoregressive
generation plus observer overhead; it excludes prompt preparation, CPU-to-GPU
handoff transfer, evaluation, upstream computation, and discarded retry attempts.
Discarded retry latency is saved separately as `receiver_retry_latency_sec`.
Checkpoints come from the final attempt. These are observed trajectory-prefix
times, rather than separately timed capped inference runs.

Let C be handed-off cache positions, P receiver prompt tokens, G generated tokens
including EOS, and D=max(G-1,0). The first token comes from prefill. Saved proxies:

- `receiver_processed_positions = P + D`.
- `receiver_attention_pairs = P*C + P*(P+1)/2 + D*(C+P) + D*(D+1)/2`.

Attention pairs count causal query/key interactions per layer/head. They account
for longer KV contexts at the same token budget and are not hardware FLOPs or a
latency estimate. Compare proxies within a model; architecture constants differ
across 4B/8B/14B. Raw records include C/P and each budget's proxy and latency.
Curves include means and paired bootstrap 95% CIs for each cost. Cost differences
are reported for matched-minus-mismatched and each handoff-minus-no-handoff.

R* is the smallest tested finite budget whose point accuracy reaches each target;
otherwise N/A. The threshold table also reports mean generated tokens and cost at
R*, plus the smallest measured mean latency/proxy among tested budgets that reach
the target and the corresponding budgets. No interpolation or monotonic smoothing
is applied. Free is reported separately from fixed finite budgets.

## Donors and pairing

Each sample/U upstream context is built exactly once. Full caches are retained in
CPU memory and cloned to the model device. No positions are removed. Mismatched
handoff uses a deterministic minimum-total-absolute-length assignment over all
bijective derangements, separately for each U. Self-assignment is prohibited.
The assignment uses the existing SciPy installation, with no dependency changes.
It replaces the length-sorted cyclic rotation that could pair length extremes.

Recipient/donor IDs and cache lengths, signed `donor_length_delta` and
`donor_length_abs_delta`, full cache measurements and upstream agent inputs are
saved. Each curve reports mean/max absolute length differences. Equal-length
matching cannot always be achieved, particularly for unique outliers or a
two-sample smoke; residual differences remain visible rather than being hidden
through KV truncation or sample removal.

Exact common sample sets, one row per cell, and the configured full Cartesian
grid are required. Violations refuse paired statistics. Accuracy and cost CIs
use the same sample-ID resample indices for all U/M/R. Bootstrap indices and
all distributions are saved. Reports contain facts and statistics without
automated interpretation or conclusions.

## MLflow artifacts and reanalysis

MLflow is the source of truth. Execution uses temporary artifact staging, uploads
under `results/`, then removes staging. Artifacts include `sample_results.jsonl`,
`resolved_config.yaml`, `summary.md`, `metrics.json`, `sample_matrix.csv`,
`budget_curves.csv`, and `bootstrap_statistics.json`. Scalar accuracy/cost means,
CIs, and diagnostics are also logged as MLflow metrics. Failures upload available
records/config and `failure.json` and mark the run FAILED.

```bash
just analyze-receiver-compute-preflight --run-id RUN_ID \
  --target-accuracies 0.5,0.7,0.9 --bootstrap-count 2000
```

Reanalysis requires a FINISHED preflight run and downloads raw records/config
from MLflow afresh. It needs no model or existing local result cache. Saved raw
text is rescored with the strict final-answer policy, including the exported
sample matrix. Existing records without measured costs produce N/A cost metrics;
latency cannot be recovered from text. Reanalysis performs no inference and
cannot extend a previously truncated trajectory. Updated endpoint retry and cost
collection require the updated execution path.

Outputs default to `artifacts/receiver_compute_preflight/RUN_ID/` and are uploaded
to the source run under `analysis/`. `MLFLOW_TRACKING_URI` follows the usual tracker
configuration; `source.json` records run provenance. Run `just lint && just test`.
