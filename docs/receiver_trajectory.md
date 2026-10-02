# Secret-digit receiver trajectory

## Collection

The collector supports Qwen3-0.6B, 4B, 8B and 14B in that order. Each model uses
100 balanced receiver samples at seed 42, sender prompt v1, receiver prompt v2
with thinking disabled, `The number is ` scoring prefix and contextual
single-token digit argmax scoring. Conditions remain `latent_only/own` and
sample-matched `drop` at receiver position zero.

Build one full20 sender KV cache per receiver sample. For each k=1..20, retain
`full[:prompt_len+k]`, then keep its final k positions. Original RoPE keys are
retained, with `original_full_seq_len = position_start = prompt_len+k`.
Taking the final k positions directly from full20 would include future states.

```bash
# Parity and steps 1..20 on ten canonical samples (one per digit), with full tracking
just run-receiver-trajectory --smoke
# Full collection on the local 8GB GPU
just run-receiver-trajectory --model 0.6B
# Larger GPU; models processed sequentially
just run-receiver-trajectory
# 14B collection only
just run-receiver-trajectory --model 14B
```

BF16-capable CUDA is required. 4B/8B/14B require at least 30 GiB total VRAM.
For 14B, that minimum is only an admission check, not a guarantee of sufficient
free memory: BF16 weights alone occupy roughly 28 GB, so 48GB gives more room.
No quantization, dtype change or offloading is introduced.

## Parity gate

Before collection, k=1 and 4 are compared on ten representatives to newly
created independent k-step rollouts. Every layer's KV tensors and receiver
next-token logits must match exactly. No override can bypass failure.

Parity uses freshly generated independent rollouts only. No prior run,
matching GPU/commit/library record, local acquisition CSV/JSONL, or probe analysis
file is required to start collection. GPU and library versions remain provenance.

## Saved data

MLflow is the source of truth. Each model uses a temporary staging directory;
artifacts are uploaded before that directory is automatically removed, including
on failure or interruption. Collection has no `--output-dir` option.
Each MLflow run contains:

- `sample_results.jsonl` and `.json`: 2100 complete `ReceiverAcquisitionRecord` observations
  (100 drop + 2000 own). Includes sample/source IDs and digits; ten candidate
  probabilities/log-probabilities; argmax, candidate mass, rank, margin and
  source/target scores; top-five vocabulary tokens; KV length/bytes/layers/dtype
  and original/retained positions; receiver prompt length; latency and errors.
  Every row also retains the actual receiver messages, rendered scoring prompt,
  input IDs/mask/tokens and a structured `raw_output` identifying digit and
  full-vocabulary argmax separately. There is no generated answer string.
- `receiver_outputs/<sample_id>.pt`: all 21 full-vocabulary next-token logit
  tensors for that sample, detached on CPU without changing their dtype.
  Each row and span links its artifact path and exact tensor key.
- `sender_contexts.jsonl`: the 100 sender samples' actual messages, rendered
  prompt, token IDs/mask/tokens, sample IDs, rollout/cache lengths and timing.
- `sender_probe_inputs.jsonl`: the 200 probe collection examples' actual messages,
  rendered prompts and tokens. The same metadata accompanies saved states.
- `sender_latent_states.pt`: the existing probe collection protocol, with 20
  prompt templates × ten digits × 20 steps. Stores `hidden_pre_realign`,
  `latent_post_realign`, digit labels, template groups and example metadata.
  These 200 examples are distinct from the balanced 100 receiver samples.
- `candidate_token_mapping.json`: token IDs, contextual candidate mapping and
  actual receiver prompt and its hash from the acquisition scoring helper.
- `receiver_input.json`: actual receiver prompt/messages, token IDs and attention mask.
- `resolved_config.json`: the collection command options and protocol settings.
- `source.json`: model, dtype/backend/device, seed/sample count, prompt versions
  and templates, answer prefix, handoff positions, collection settings,
  Git commit, GPU and library versions.
- `parity.json`: exact KV/logit/position checks against fresh independent rollouts.

Collection loads the existing `.env` tracking URI and uploads these artifacts
and collection counts to the separate `latentmas_receiver_trajectory` MLflow
experiment. It does not fit probes, run permutations/bootstrap, join probe
results, aggregate receiver statistics or generate a scatter.

The original acquisition preset's optional receiver hidden states and raw KV
were disabled; they remain disabled here. Its other carrier/condition cells
(`full`, `prompt_only`, `cross`, `drop_position_matched`) are not collected by this own/drop trajectory experiment.

Each receiver sample has a `secret_digit_sample` MLflow trace, with one sender
context span and 21 receiver forward spans (drop plus steps 1..20). Span inputs
identify the step and condition; outputs include digit scores and cache/position
metadata. Sender span inputs include the actual sender prompt/messages and input tokens.
Receiver span inputs include the exact scoring prompt/messages and input tokens;
outputs retain structured raw predictions plus links to the full logits artifact.
Root outputs retain all 21 observations without overwriting steps.
Target-digit expectations, source-follow feedback and execution status follow
the existing acquisition tracker. Traces are flushed before the run ends.
Runs completed before this tracing fix have artifacts but no sample traces;
real forward traces require a new execution.

Previously completed 0.6B artifacts predate sender-state collection; that run
contains receiver observations and probe-joined summaries, but not the newly
added sender-state artifact. No historical run is modified or backfilled.

## Offline analysis

Analyze the saved sender states of each completed sweep directly from MLflow:

```bash
just analyze-sender-probe --source-run-id SWEEP_RUN_ID --backend torch --permutations 5000
```

The command detects the `latentmas_receiver_trajectory` source experiment,
reads that sweep's root `sender_latent_states.pt` and `source.json`, and creates
its probe analysis in the same experiment. The original collection run is
preserved. The new run has `phase=sender_probe`, an exact `source_run_id`,
`source_artifact=sender_latent_states.pt`, model and latent-step metadata.
All 20 steps are analyzed without loading a language model or running inference.
Missing states are an error; no acquisition run or local file is substituted.
Local config/state overrides and `--replace-source-run` are rejected for sweeps.
Probe inputs must contain the full 200 examples and 20 latent steps.

For the completed 14B sweep:

```bash
just analyze-sender-probe \
  --source-run-id 12fe75f4f0cc44c09c19ed566d40b283 \
  --backend torch --permutations 5000
```

Run the same command for the 0.6B, 4B and 8B sweep IDs. Use the resulting probe
run IDs with `just analyze-stepwise-communication`, described below. A probe
from an acquisition run, an unselected sweep, an incorrect artifact path, or a
duplicate probe for the same sweep is rejected before the communication plots
are generated. Existing acquisition probes do not qualify as sweep probes.

## Tracking coverage

The collection preserves the original secret-digit tracking fields, restricted
only to the requested latent-only own/drop conditions. Raw data is kept for
later analysis; fitting and aggregate statistics stay outside collection.

| Original tracking responsibility | Trajectory location |
| --- | --- |
| Model/task/method, sample count/seed, backend/dtype/device, prompt templates/versions, carrier/condition settings, state-saving flags, Git revision | MLflow tags/params plus `resolved_config.json` and `source.json` |
| Contextual digit token mapping and tokenizer diagnostics | `candidate_token_mapping.json` |
| Sample/source identity, digit probabilities/log probabilities, argmax/mass, source/target probability/rank/margin/correctness, top-five tokens | Every receiver span and full observation JSON/JSONL |
| KV length/bytes/layers/dtype, original/retained/receiver positions, prompt lengths, latency and errors | Full observation JSON/JSONL, sender context artifact and spans |
| Sample root, sender context and receiver forward tracing | 100 sample traces; one sender span and 21 receiver spans each |
| Target expectation, source-follow feedback and execution status | Trace assessments and error status |
| Pre/post sender states, digit labels, template groups and example metadata | `sender_latent_states.pt` plus `sender_probe_inputs.jsonl` |
| Failure details | Error spans/observations plus `failure.json` and failure traceback parameter |
| Probe fitting, permutation/bootstrap and receiver aggregate statistics | Deferred analysis commands |

Actual prompt text and full receiver logits are also recorded explicitly.
Existing completed runs lack any data they did not originally capture; missing
full logits or real trace contents cannot be invented retrospectively.

## Strict three-model communication analysis

`just analyze-stepwise-communication` reads only MLflow run artifacts for
Qwen3-0.6B, 4B and 8B, with steps 1..20. MLflow is the sole source of truth;
there are no local CSV input options, persistent input cache, or local fallback.
`MLFLOW_TRACKING_URI` (loaded from `.env`) or `--tracking-uri` is required.
Downloads are temporary and removed after the saved results have been read.
No model loading, sender/receiver inference, probe fitting, permutation runs,
regression, smoothing or pooled correlation is performed.

Explicit run IDs prevent silently choosing the latest of multiple experiments:

```bash
just analyze-stepwise-communication \
  --probe-run-ids PROBE_06B PROBE_4B PROBE_8B \
  --sweep-run-ids SWEEP_06B SWEEP_4B SWEEP_8B \
  --independent-run-ids OWN_06B_1 OWN_06B_4 OWN_06B_20 \
    OWN_4B_1 OWN_4B_4 OWN_4B_20 OWN_8B_1 OWN_8B_4 OWN_8B_20
```

Inputs must be FINISHED runs in their respective MLflow experiments:

- Three `latentmas_receiver_trajectory` probe runs with `phase=sender_probe`,
  each referencing exactly one of the selected sweep run IDs and its root
  `sender_latent_states.pt`. `probe/results.json` supplies observed accuracy
  and FWER p-value;
  `probe/null_statistics.json` supplies the saved null accuracies. Only
  `latent_post_realign` cells are used, with the mean of each saved null column.
- Three `latentmas_receiver_trajectory` runs with `sample_results.jsonl`,
  `source.json` and passing `parity.json`. All 100 canonical samples must have
  paired drop and latent-only own observations at every step 1..20.
- Nine independent `latentmas_receiver_acquisition` runs with
  `results/sample_results.jsonl`, at steps 1, 4 and 20 for each model. The
  latent-only own predictions are compared against their sample-matched drop
  observations; other acquisition conditions are not used in this metric.

Missing/duplicate model-step cells, incomplete samples, invalid identities or
handoff positions, missing metrics and nonfinite values raise errors. No cells
are silently dropped or pooled. Exactly 60 final metric rows and nine parity
rows are required. Parity compares the changed fraction used in the scatter;
it does not claim full-logit equality against the historical independent runs.
A mismatch saves the comparison tables to a FAILED analysis run before raising
an error; plots are generated only when all nine metrics match exactly.

Derived artifacts are uploaded under `communication/` in a new
`latentmas_receiver_trajectory` run tagged `phase=analysis`:

- `stepwise_communication_metrics.csv`: the exact 60 rows used by every plot,
  including effect in percentage points, receiver changed percentage and
  significance derived directly from FWER p < 0.05.
- `stepwise_receiver_parity.csv`: independent/sweep changed fractions,
  differences, run IDs and equality flags at steps 1, 4 and 20.
- `stepwise_communication_scatter.png` / `.pdf`: model colors, thin connections
  in step order, filled significant points and hollow nonsignificant points.
  No default step labels; `--annotate-steps` labels only 1, 4 and 20.
- `stepwise_communication_diagnostics.png` / `.pdf`: per-model unsmoothed
  step curves for probe effect and receiver changed percentage. All model panels
  share step limits/ticks; probe effect and receiver change use the same
  measurement limits/ticks as the scatter. Probe limits span all 60 effects
  with 2 pp ticks; receiver ticks are 0..100% in 20% increments.
- `source_runs.json`: all input run IDs and artifact paths for reproducibility.

`--output-dir` optionally exports copies of these derived artifacts locally;
those files are never analysis inputs. The 60 points are not treated as iid
observations. The figure shows whether the two trajectories separate under
different conditions, without a test claiming absence of correlation.

Earlier communication analysis runs used acquisition probes paired with sweep
receiver observations. Their receiver parity checks do not validate sender
provenance, and those figures are not a same-sweep analysis. The current
communication command rejects those inputs; regenerate each sender probe from
its selected sweep before producing the communication figures.
