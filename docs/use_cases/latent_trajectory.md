# Training-Free Latent Computation Trajectory

Independent frozen-model pilot experiment on GSM8K with Qwen3-8B. It does not
assume that latent computation improves reasoning. No training, probes or patching.

```bash
just run-latent-trajectory -c latent_trajectory/gsm8k_q38 --dry-run
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/gsm8k_q38
# The collector supports any sample count; one sample is only the preset limit.
just run-latent-trajectory -c latent_trajectory/gsm8k_q38 --max_samples 2
just analyze-latent-trajectory storage/experiments/latent_trajectory/gsm8k/Qwen3-8B/RUN_HASH
```

The preset uses the existing direct_solve R(Q) messages, seed 42 and 2048-token
continuation budget. It preserves the actual existing 8B intervention sampling
policy: temperature 0.6, effective top-p 1.0 and top-k 0, tokenizer EOS termination.
Requested top-p 0.95 and effective settings are both recorded. Latent realignment
uses the existing model adapter and is disabled by default, as in that preset.

| Condition | Path | Shared trajectory |
|---|---|---|
| a | R(Q) → Lk → E | without_think |
| b | R(Q) → <think> → Lk → E | with_think |
| c | R(Q) → Lk → </think> → E | without_think |
| d | R(Q) → <think> → Lk → </think> → E | with_think |

`latent_steps: 80` is the trajectory endpoint; `latent_step_start: 1` is the first
readout boundary. All preceding latent steps are still calculated and saved.
`--conditions a c` selects readouts without creating unnecessary trajectories.
The assistant header is rendered with the tokenizer's thinking-enabled template,
then only its automatic opening-think suffix is removed. Prompt message contents
are preserved. Unknown suffix formats fail rather than silently changing inputs.
Thinking markers must encode to distinct native single token IDs. Qwen3 labels
these added vocabulary tokens with `special: false`; they are inserted by their
existing IDs without changing the tokenizer. Condition c deliberately
retains its unmatched closing tag; it is never repaired.

The existing intervention runner performs each trajectory once and returns its
complete prompt-plus-80-step KV and hidden states. Both selected trajectories are
saved before any explicit decoding. A readout copies the saved cache prefix
`[0:prompt_length + k]`, including the entire prompt and the first k latent
positions. The with-think trajectory's prompt length includes its opening token.
Closing-tag readouts prefill the
actual closing token before decoding. No new chat message is inserted at readout.
The latent hidden is the final-layer output *after* the kth latent forward;
raw logits are LMHead(hidden), before sampling transformations or closing tags.
Every readout restores a sample-derived RNG seed in an isolated RNG context.
Condition order and skipping finished continuations do not consume trajectory RNG.

Artifacts are isolated under
`storage/experiments/latent_trajectory/<task>/<model>/<config-hash>/`:

- `metadata.json`: model/tokenizer revisions, dtype, template, protocol and decoding.
- `samples/<dataset-index>/sample.json`: question, gold, role messages, exact base
  prompt and IDs, thinking IDs and sample seed.
- `samples/<dataset-index>/trajectories/{without,with}_think.pt`: steps, actual
  prefill IDs, hidden `[80, hidden_dim]` and raw FP32 logits `[80, vocab_size]`.
- `samples/<dataset-index>/trajectories/{without,with}_think/kv.pt`: complete CPU
  KV cache, prompt length and latent step count. Prefix slices are independent;
  the saved full trajectory cannot be changed by continuation decoding.
- `samples/<dataset-index>/continuations/<condition>/step_NNN.json`: all generated
  IDs (including sampled terminal EOS), raw text with special tokens, evaluation
  text, extracted answer, correctness, lengths, termination and generated tag
  positions (zero-based within the saved generated IDs). Inserted tags are separate.

`generated_length` includes terminal EOS; `content_length` follows the existing
intervention convention and excludes it. No artificial EOS is added on budget
exhaustion. Evaluation uses the existing special-token-stripped text convention.

Continuations are saved atomically as each branch completes. Persisted full
trajectories are reused even when readouts are missing: only unfinished readouts
are decoded, with no prompt or latent forward calls. An interruption before a
trajectory's KV is saved requires recomputing that trajectory. Existing completed
continuations are retained. If older results contain hidden/logits without KV,
the trajectory is computed once, checked for exact equality with the saved
tensors, and its KV is saved for subsequent reuse. Sample limits and selected conditions do not change the
run identity. Use a separate output root to request an independent repeat.

The analysis command loads only saved tensors and writes `.analysis.json` next to
each trajectory: L2 norm, adjacent cosine/L2, successive displacement cosine,
entropy (nats), top-10 token IDs/probabilities and adjacent Jensen–Shannon divergence
(natural logarithms). Missing predecessor metrics are null. Token IDs can be mapped
using the recorded tokenizer; no model is required. Continuation records retain all
inputs needed for subsequent correctness, length, reasoning and paired-condition
comparisons. No visualization system is included.

Default completion is two 80-step hidden/logit artifacts, two full KV artifacts,
and 320 continuation records
per sample. Unit tests use a small model; a full Qwen3-8B run remains a separate GPU
execution, and is not part of the unit suite.

## Qwen3-14B and MLflow

The default 14B presets execute alignment OFF then ON. GSM8K selects indices
`[0, 660]`; MedQA selects `[0, 145]`. Each `(alignment, sample)` is one independent MLflow run, giving four runs per
dataset preset. The model is loaded once per alignment and reused across samples. Both presets retain 80 latent steps and A/B/C/D
readouts. The 8B GSM8K preset still defaults to one sample and alignment OFF.

```bash
just run-latent-trajectory -c latent_trajectory/gsm8k_q314 --dry-run
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/gsm8k_q314
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/medqa_q314
```

After both trajectories are saved, decoding runs in step order: A, B, C, D at
k=1, then A, B, C, D at k=2, through k=80. Selected subsets retain this condition
order. Finished local continuations are skipped without changing this schedule.

Collection uses the shared `MLflowTracker`, honoring `MLFLOW_TRACKING_URI` and
its existing local-storage defaults. The experiment is
`latentmas/latent/trajectory`. Each run records protocol parameters, requested
conditions/sample limit, resolved model/decoding metadata, per-sample and
per-condition stepwise correctness/length metrics, and all local artifacts,
including raw hidden/logit tensors. `samples/<index>/tracking.json` links that sample to its MLflow run. The configured GPU must accommodate the chosen model and KV cache.

After each step's requested conditions finish, a single background worker uploads
one `(sample, latent_step)` trace while decoding proceeds to the next step. With
the default four conditions, each trace has four decode spans (A/B/C/D), for 80
traces per sample. Root inputs/tags include the sample ID and latent step.
Like intervention traces, the response preview summarizes each condition's
prediction and PASS/FAIL. Root outputs expose `a_prediction`, `a_correct`, etc.;
trace assessments record `expected_answer` and boolean `a_correct` through
`d_correct` feedback for selected conditions. Run metrics alone do not supply
these trace-level evaluation columns.
`recording=saved_results` makes clear that span
durations measure recording rather than historical inference latency. Existing
results skipped during inference are included. Interrupted collection records a
FAILED run, failure details and available partial results. Tracking/upload errors
propagate and leave local results intact. Repeated invocations may add traces to
the same run under the shared tracker's run-reuse policy.

Tracking reuses the intervention tracker's `start_staging_run`,
`promote_staging_run` and `discard_staging_run` lifecycle and the same single-worker
executor pattern. Before promotion, all background trace tasks must succeed;
the shared promotion flushes traces and links them to the canonical final run.
Artifacts are uploaded during final promotion, and per-step metric histories are
copied to the final run. Completed steps found during shutdown are also queued;
a step missing any requested condition is not published as a partial trace.
Its completed local continuations remain available for resumption. Upload errors
propagate while local inference files remain.
No persisted inference identity or tensor format changes with this tracking update.
To adopt it, stop the old process with Ctrl-C and rerun the same command. A hard
kill preserves atomically completed files but cannot finish MLflow shutdown.

## Intervention recording contract

Trace rendering now directly calls the existing intervention trace renderer.
Each root is named `sample_<dataset-index>_step_<k>`, with `initial_states` for
the two prompt/latent trajectories and `decode_a` through `decode_d` children.
Only selected conditions are emitted. The same renderer supplies task/model/seed/
git tags, role and prompt inputs, intervention text/IDs, effective sampling,
generation budget, RNG provenance, token usage, answer expectations, correctness
feedback, root prediction/correctness outputs and the PASS/FAIL response preview.
The `latent_step` tag and `dataset_sample_key` distinguish step and dataset index.
In trace views, generated text uses the existing intervention evaluation-text
convention and generated lengths/token usage exclude terminal EOS. The original
continuation JSON still preserves raw text and all generated IDs including EOS.

The existing intervention search and span verification helpers are reused.
A fingerprint identifies each sample/step's inputs and condition results. A valid
unchanged trace is linked/reused; missing or changed traces are created and fetched
back for verification before previous traces for that identity are deleted.
Local `samples/<sample>/traces/step_NNN.json` records the validated trace reference.

Each continuation also stores `continuations/<condition>/step_NNN/hidden.pt`
using intervention's hidden alignment: states predicting generated non-EOS tokens,
with the prefilled closing-token state retained for C/D even on immediate EOS.
The raw record includes completion status, artifact list, initial role/prompt/IDs,
intervention details, RNG provenance and the shared decoder's result identity.
Each trajectory additionally saves `trajectories/<name>/prompt_hidden.pt`.
`state.json` and metadata record the existing intervention state identity and
execution provenance. The shared summary writer produces `summary.json` and
`samples.jsonl`, including per-condition accuracy/completion and trace references.

Old continuation JSON files alone do not contain the missing generated hidden
states. Resumption replays those readouts with the original sample seed, requires
exact equality of the previously saved token sequence, and then saves the missing
hidden/provenance records. A mismatch raises an error and leaves the prior result
intact. This backfill requires decoding time but does not rerun latent computation
when full KV is already saved. Missing prompt hidden requires only prompt prefill.
Fully populated records continue to be skipped.

During continuation collection there are three tqdm bars: sample completion for
the current alignment (position 0), latent-step completion for the current sample
(position 1), and generated tokens for the current condition (position 2). A step
advances only after all requested conditions finish. A sample advances after its
tracking context completes. Saved complete samples/steps initialize the bars;
partially completed work advances when its remaining calculations finish. Fixed
`Decoding sample=...` log lines are not emitted. `--no_progress` hides all bars.

## Selecting dataset indices

Without `sample_indices`, `max_samples: 1` selects index 0, a positive N selects
the first N samples, and a non-positive limit selects all samples. The 14B presets
explicitly select their two sample indices; the 8B preset retains the first-sample
default. To select specific GSM8K test indices (zero-based):

```bash
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/gsm8k_q314 \
  --sample_indices 0 12 37
```

Explicit `--sample_indices` overrides `max_samples`, including the preset's limit
of 1. Duplicates are removed and execution follows original dataset order. The same
list can be supplied as `sample_indices: [0, 12, 37]` in YAML. Negative or missing
dataset indices are rejected; an out-of-range selection fails before collecting
any of its samples.

Sample directories, seed derivation, step callbacks, trace identities and summary
indices always use the original dataset index (`000012` for index 12), never the
position within the selected list. Each selected sample retains k=1..80 and
A/B/C/D ordering with the three progress bars. Selection does not change the run or
cache directory identity, so matching saved KV/results remain reusable. Invocation
metadata records the requested selection; shutdown trace publication and current
summary metrics are restricted to that selection. Other local sample results
remain available for later runs.

## Latent alignment toggle

The 8B preset defaults to `latent_space_realign: false`; the 14B presets use
`[false, true]` to run both settings. CLI flags explicitly override the YAML value:

```bash
# ON
just run-latent-trajectory -c latent_trajectory/gsm8k_q314 --latent_space_realign
# OFF, including when a custom YAML preset enables alignment
just run-latent-trajectory -c latent_trajectory/gsm8k_q314 --no_latent_space_realign
```

The flags are mutually exclusive and can be combined with `--sample_indices`.
Alignment remains part of the run/cache identity, so ON and OFF trajectories and
results are stored separately. The existing model adapter supplies both policies:
OFF disables the realignment matrix, while the existing input-embedding norm
normalization remains active.

To run OFF and ON as separate runs from one configuration, use a boolean list:

```yaml
latent_space_realign: [false, true]
```

A ready-to-run 14B preset is included:

```bash
just run-latent-trajectory -c latent_trajectory/gsm8k_q314 --dry-run
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory \
  -c latent_trajectory/gsm8k_q314 --sample_indices 0 12 37
```

This executes the OFF sample batch first, then ON, with separate alignment cache
directories and one MLflow run per sample. Each sample run uses the same latent
steps and A/B/C/D conditions.
Only the alignment field is expanded; condition and sample-index lists remain
within each run. Single booleans retain their existing one-run behavior. An
explicit CLI ON/OFF flag overrides the list and selects one run. Empty lists and
non-boolean values are rejected. Repeated boolean values are executed only once.
The dry-run output lists both fully resolved configurations and their paths.


MedQA uses the existing `data/medqa.json` dataset loader in its original file
order. Its `single_agent` role reuses the task-aware prompt requiring a boxed
A/B/C/D answer, and the existing evaluator normalizes and checks that letter.
No dataset/model inference is performed by preset dry runs.


## Per-sample runs and progress

MLflow run identity is `(configuration, dataset sample index)`. Display names are
`<task>_<model>_align-<on|off>_sample-<index>_<configuration-hash>`; the hash keeps
different decoding/configuration runs distinct. The defaults produce four runs
per dataset and eight across GSM8K plus MedQA. Every sample run contains 80 step
traces with four condition spans each. The async worker publishes each completed
step immediately to its sample's staging run, which is promoted independently.

The tensor/continuation cache root remains unchanged. Existing fully populated
results can be reused without inference. The trace fingerprint and search group
also remain unchanged, so validated historical traces can be linked to the new
sample run. Historical multi-sample runs are retained; they are not overwritten.
New runs use the sample directory for tracking identity, so no two samples share
a run. Run names do not change the inference cache identity.

Each sample's `tracking.json`, `summary.json`, `samples.jsonl`, `invocation.json`
and any `failure.json` now live under `samples/<index>/`. Its MLflow artifact root
contains only that sample's files plus the common `metadata.json`/`state.json`.
Trace creation, verification, deduplication, step metrics and condition feedback
retain the intervention recording contract. A failed sample preserves local
results and fails its own run; later samples are not started after that failure.

Example while collecting the default GSM8K OFF batch:

```text
Samples · alignment=OFF          1/2
Sample 000660 · step 28/80      27/80
Sample 000660 · B · latent 28/80   512/2048 token
```

Dry runs list the per-sample `mlflow_runs` names and indices within each alignment
plan. If selecting the full dataset without an explicit positive limit, sample
runs are resolved from the dataset during collection rather than during dry run.
