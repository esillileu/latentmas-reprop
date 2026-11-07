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

`latent_steps: 80` is the trajectory endpoint; `latent_step_start: 0` is the first
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

## Offline partial-result analysis

```bash
just analyze-latent-trajectory storage/experiments/latent_trajectory/gsm8k/Qwen3-14B/RUN_HASH --continuations-only
# No tokenizer or model weights required for correctness/length/boundary analysis:
just analyze-latent-trajectory SOURCE --continuations-only --skip-token-distance
# Pair an opposite-alignment collection at the same condition and step:
just analyze-latent-trajectory SOURCE --compare-directory OTHER_SOURCE --continuations-only
```

Analysis snapshots available continuation paths and uses the collector's completion
contract (complete status, hidden artifact, RNG provenance and decoder identity).
Missing, incomplete and invalid observations remain explicit rows; none are counted
as incorrect. Missing conditions and interrupted collection do not prevent reports.
Malformed continuation JSON or inconsistent record identity is reported as invalid.
Unavailable correctness cells are gray; curves break at unavailable steps.

Outputs under `artifacts/latent_trajectory/<task>/<model>/<config-hash>/`
(or `--output-dir`) include:

- `readouts.csv`: availability, evaluator correctness, prediction, lengths, finish
  reason, budget exhaustion, generated/prefilled thinking token counts and
  explicit thinking-boundary token lengths.
- `trajectories.csv`: within-item correct-step fractions and adjacent correctness
  changes. These fractions are not accuracy over independent items.
- `step_accuracy.csv`: across-item step accuracy with available item IDs and denominators.
- `condition_comparisons.csv`: paired same-step contingency tables and accuracy
  differences; unavailable comparisons have empty values and zero pairs.
- `alignment_comparisons.csv` when requested: protocol-checked paired comparisons.
  Different inference protocols or question/gold mismatches are marked incompatible.
- `review_candidates.json`: full before/after outputs for adjacent correctness
  changes, with semantic judgments left unreviewed. Human annotations belong in a
  separate file; generated candidates may be replaced when analysis is rerun.
- `sample_<index>.png` and `summary.md`: availability/correctness heatmaps,
  content/thinking lengths, adjacent distances and transition review excerpts.
- `source.json`: source metadata and feature-availability notes.

All complete readouts, including B boxed-only answers, contribute to correctness
and adjacent distances. Distances exclude terminal EOS using `content_length`,
collapse Unicode whitespace, preserve symbols/thinking tags and retokenize with
`add_special_tokens=False`. Unit-cost Levenshtein is divided by maximum token
length (empty/empty = 0). Gaps are never bridged. If the recorded local tokenizer
is missing or has a different identity, token distance is unavailable while other
analyses proceed. No downloads or new inference are performed.

### Evaluator audit and optional human review

Saved evaluator judgments are trusted by default. With `--audit-evaluations`,
`evaluation_audit.csv` records an explicit replay of each saved `evaluation_text`
and compares it with the recorded prediction and correctness. Missing evaluation text is unavailable, not an automatic disagreement.
This verifies reproducibility of the evaluation path, not semantic correctness.

For MedQA, `gold_audit.json` checks the saved question and gold against local
`data/medqa.json` (override with `--medqa-data`). It checks exact option text and
flags ambiguous substring mapping. Conservative explicit final-choice detection
flags possible extraction disagreements without replacing stored scores. Source
labels are taken from the dataset; this audit makes no medical judgment.

`review_template.json` contains unique before/after readouts at correctness
transitions. Copy it to a separate file and fill only judgments you have reviewed:

```bash
cp artifacts/latent_trajectory/TASK/MODEL/RUN/review_template.json /path/to/annotations.json
just analyze-latent-trajectory SOURCE --continuations-only --annotations /path/to/annotations.json
```

Each entry identifies `source_run_id`, integer `sample_id`, `condition` and integer
`step`. Optional judgments are `semantic_correct` (boolean), `reasoning_validity`
(`valid`, `invalid`, `no_reasoning`, `undetermined`), `restart` (boolean), and human
text in `reason_label`, `reasoning_progress`, `review_notes`. Null fields remain
unreviewed; evaluator correctness is retained as a separate field. Duplicate keys,
invalid values and unavailable identities are reported in `annotation_status.csv`
without applying them. Source/gold mismatches are not silently merged. Reviewed
fields are written to `readouts.csv` and review status appears in the case report.
Generated candidates/templates can be refreshed, while supplied annotation files
are preserved, including when the template itself is supplied as input.

Thinking-boundary lengths count only tokens within explicit opening/closing
markers, considering both prompt and readout prefill. C/D begin after a closing
prefill; generated re-entry is counted separately through the same token scan.
Zero boundary tokens does not establish absence of semantic reasoning. Evaluator
correctness does not establish valid medical reasoning; all semantic labels start
unreviewed and change only through supplied human annotations. Paired uncertainty
is reported as exploratory pointwise intervals for selected items; see below.

Without `--continuations-only`, saved shared hidden/logit tensors produce:

- `latent_features.csv`: one row per source/sample/shared trajectory/step, with
  separate hidden/logits/geometry availability. Shared states are extracted once.
- `readout_latent_features.csv`: A/C join without-think, B/D join with-think;
  correctness/availability and human annotations retain readout identities.
- `latent_events.csv`: adjacent correctness changes with current and preceding
  state metrics. Duplicate shared states are not independent statistical samples.
- `latent_summary.md`, per-trajectory time series and geometry PNGs: feature curves,
  readout correctness, centered PCA and adjacent hidden-L2 versus logit-JS scatter.
- Per-trajectory `.analysis.json` and `tensor_status.json`, all in the output tree.

Hidden features include norm/mean/std/max absolute value, adjacent cosine/L2,
step-0 distance/cosine and displacement-direction cosine. Step 0 is explicitly
`prompt_hidden[-1]`; its absence leaves origin metrics unavailable while latent
metrics continue. Centered PCA and singular-value-entropy effective rank are fitted
once per shared trajectory using available states (including step 0 when present).
Axes are local to each fit and must not be compared directly across runs.

The main condition-comparison view is `common_pca_summary.md`, generated for both
GSM8K and MedQA by the same default command. For each model/item it fits one mean
and PCA basis over available OFF/ON × without/with-think states. A/C and B/D are
included once as shared trajectories. Each trajectory has equal total weight,
with uniform weights within it, so unequal saved step counts do not dominate.
All states use the same mean and basis, without individual recentering.

The opposite-alignment source is discovered among sibling directories only when
exactly one matches protocol/tokenizer checks. Select it explicitly with
`--compare-directory OTHER_SOURCE`. Different tasks, models, protocols, questions,
golds or base prompt IDs are not silently pooled. Missing/incompatible groups
remain gray in the legend while available states produce a partial fit.

`common_pca_coordinates.csv` stores coordinates, `common_pca_status.json` records
source membership/availability and explained variance, and
`common_pca_<sample>.pt` stores the mean/components for exact reprojection. Plots
show PC1/PC2 and combined explained variance, start/end and step markers.
The basis is local to one item and may change when more trajectories arrive.
Coordinates from different items, benchmarks or report generations are not
directly comparable. Individual PCA remains a supplementary within-trajectory view.

### Global PCA across all saved items

```bash
just analyze-global-latent-pca --model Qwen/Qwen3-14B
# Omit --model to fit each saved model separately.
```

This explicitly fits one common mean/basis across all saved GSM8K and MedQA items
of the selected model. It trusts the saved state as-is: historical runs remain
included, with no freshness checks, current-run filtering or automatic refresh.
Fit weights balance benchmarks, items within each benchmark, saved shared
trajectories within each item, and states within each trajectory. A/C and B/D
remain single shared trajectories. Different models are fitted separately.

The report under `artifacts/latent_trajectory/global_pca/<model-name>/` contains
an overall overlay and benchmark panels using identical axes and limits,
`global_pca_coordinates.csv`, `global_pca_status.json`, and `global_pca.pt` with
the common mean/components. Axis labels show explained variance. Missing hidden
states remain unavailable and do not prevent fitting the available states.
This global view supplements the item-specific common PCA: between-item or
benchmark variation may dominate the global components.

### Output instability diagnostics (B7)

The existing trajectory analysis also writes `output_stability.csv`,
`output_stability_summary.md`, and per-item PNGs. These include repeated content
4-gram fraction, longest identical-token run, exact normalized-line repetition,
Unicode letter-script profiles/switches, generated chat-opening markers and odd
line-leading code-fence counts. Terminal EOS is excluded from token repetition;
structural chat/thinking tags are excluded from script counts. These are
observable diagnostics, not a language detector or automatic incoherence verdict.
Missing values remain gaps. No classifier, model inference or LLM judge is used.

### Saved KV structure (E1–E4)

```bash
just analyze-latent-kv SOURCE
just analyze-latent-kv SOURCE --samples 1288 --layers 0 10 20 30 39
```

This separate command reads one saved shared trajectory cache at a time without
cloning/mutating full KV or running a model. It writes to the collection's
`artifacts/latent_trajectory/<task>/<model>/<run>/kv_analysis/` directory:

- `kv_layers.csv`: per-layer K/V prompt/latent means, standard deviations, norm
  means, finite-position counts, uncentered numerical/effective rank and leading
  singular-value energy fraction for **both** prompt and latent regions.
- `kv_positions.csv`: every prompt/latent position with region, cache position,
  region position, norm, adjacent cosine/L2, prompt-last/centroid cosine and nearest
  prompt position/cosine. This replaces the former latent-only `kv_steps.csv`.
- `kv_status.json`, `summary.md`, full-cache boundary heatmaps, separate prompt vs
  latent heatmaps with shared color ranges, and region rank/energy comparisons. Missing files,
  invalid layers and nonfinite positions remain unavailable/gray; adjacent metrics
  do not bridge invalid positions.

Step 0 is the last prompt cache position; latent step k is at prompt_length+k-1.
Earlier prompt positions have an empty step and their own zero-based prompt index.
Full-cache heatmaps mark the prompt/latent boundary in red. Prompt nearest-neighbor
queries exclude their own position, so the prompt baseline is not trivially 1.
Latent queries search all valid, nonzero prompt positions.

Rank curves include full-region values and an equal-length control: the last W
prompt positions vs the first W latent positions, W=min(prompt_length, latent_steps).
The matched valid-position counts are also saved. This helps distinguish region
length effects from differences within equal-length windows; no subsampling,
padding or new model forward is used.

Vectors concatenate batch/heads per sequence position. Saved K retains positional
encoding/RoPE, so key similarity can reflect position. KV rank uses uncentered
finite rows and is distinct from centered hidden PCA rank. Prompt cosine
does not measure attention or causal usefulness. Attention collection (E5) and
causal interventions (F2–F6) use the separate explicit causal command below.

Raw-logit features include entropy, top-10 IDs/probabilities (token strings when
the recorded tokenizer is available), concentration, adjacent KL(current||previous)
and JS, JS from saved latent step 1, and thinking-marker logits/probabilities.
Step 0 logits were not saved and are left unavailable; step 1 is explicitly the
first saved logit reference, not a replacement for step 0. C/D share the same raw
trajectory logits as A/B respectively; C/D actual post-closing-prefill start logits
require LM-head weights. The separate weight command below supplies actual-start
distributions and injected-vector/input-embedding comparisons; the causal command
supplies conditional answer scoring. The lightweight saved-tensor analysis does
not load model weights or perform those forward passes.

Missing/corrupt components do not suppress the other component. Tensor dimensions,
step identities and finiteness are checked. Plots show unavailable data in gray or
as gaps; no model inference or KV loading is performed. These descriptive links
cannot establish latent reasoning or causality.
MLflow report logging is opt-in with `--track`.

Default completion is two 80-step hidden/logit artifacts, two full KV artifacts,
and 324 continuation records
per sample. Unit tests use a small model; a full Qwen3-8B run remains a separate GPU
execution, and is not part of the unit suite.

## Qwen3-14B and MLflow

The default 14B presets execute alignment OFF then ON. GSM8K selects indices
`[0, 1288]`; MedQA selects `[0, 145]`. Each `(alignment, sample)` is one independent MLflow run, giving four runs per
dataset preset. The model is loaded once per alignment and reused across samples. Both presets retain 80 latent steps and A/B/C/D
readouts. The 8B GSM8K preset still defaults to one sample and alignment OFF.

```bash
just run-latent-trajectory -c latent_trajectory/gsm8k_q314 --dry-run
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/gsm8k_q314
CUDA_VISIBLE_DEVICES=3 just run-latent-trajectory -c latent_trajectory/medqa_q314
```

After both trajectories are saved, decoding runs in step order: A, B, C, D at
k=0, then A, B, C, D at k=1, through k=80. Selected subsets retain this condition
order. Finished local continuations are skipped without changing this schedule.

Collection uses the shared `MLflowTracker`, honoring `MLFLOW_TRACKING_URI` and
its existing local-storage defaults. The experiment is
`latentmas/latent/trajectory`. Each run records protocol parameters, requested
conditions/sample limit, resolved model/decoding metadata, per-sample and
per-condition stepwise correctness/length metrics, and all local artifacts,
including raw hidden/logit tensors. `samples/<index>/tracking.json` links that sample to its MLflow run. The configured GPU must accommodate the chosen model and KV cache.

After each step's requested conditions finish, a single background worker uploads
one `(sample, latent_step)` trace while decoding proceeds to the next step. With
the default four conditions, each trace has four decode spans (A/B/C/D), for 81
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
position within the selected list. Each selected sample retains k=0..80 and
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
per dataset and eight across GSM8K plus MedQA. Every sample run contains 81 step
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

## Adding step zero to an existing run

The presets now collect readouts at steps 0 through 80. Rerun the original
command with the same model, decoding settings, alignment and sample selection.
Step 0 and step 1 starts share the original cache directory and MLflow run identity.
Completed steps 1–80 are skipped. Step 0 slices the saved KV to the prompt length
and uses the last saved prompt hidden state (including `<think>` for B/D); latent
computation is not repeated when the trajectory artifacts are complete. Only the
missing A/B/C/D readouts require decoding. Raw latent hidden/logit tensors still
represent steps 1–80.

Tracking verifies/reuses existing traces, publishes step 0, records all step
metrics again, recomputes summaries, and uploads all artifacts to the canonical
sample run. MLflow parameters are immutable: an existing `latent_step_start=1`
parameter remains, with the current value `0` recorded as a tag and in metadata.
Use `--latent_step_start 1` to retain the previous readout range.


ARC-Challenge and MBPP-Plus also have Qwen3-14B presets with alignment OFF/ON,
80 latent steps and A/B/C/D readouts:

| Preset | Easy ID | Hard ID |
| --- | --- | --- |
| `latent_trajectory/arc_challenge_q314` | `MCAS_2006_9_1` | `AKDE&ED_2008_8_3` |
| `latent_trajectory/mbppplus_q314` | `127` | `239` |

```bash
just run-latent-trajectory -c latent_trajectory/arc_challenge_q314 --dry-run
just run-latent-trajectory -c latent_trajectory/mbppplus_q314 --dry-run
```

`sample_ids` (or `--sample_ids ID ...`) selects original dataset IDs and overrides
`max_samples`. Selection preserves dataset order and original numeric cache/run
indices; missing IDs raise an error. CLI `--sample_indices` replaces preset IDs,
and CLI `--sample_ids` replaces preset indices. The two options are mutually
exclusive. ID-based dry runs show requested IDs; numeric indices and MLflow run
names are resolved during collection. Both presets use the existing task-aware
`single_agent` prompts and evaluators.


MLflow publication errors (including exhausted retries during setup, step upload,
or final promotion) disable further uploads for the current CLI invocation,
including subsequent samples and alignments. Computation and local tensor/readout
saving continue. The error is logged and saved as `upload_failure.json` in the
sample directory; inference errors and interrupts still stop computation.
Rerun the same command to retry publication from saved results.

MBPP-Plus uses its source integer task IDs (`127`, `239`), written as quoted
strings in `sample_ids`. Selection compares their string representations;
loaders preserve the source ID values.

### Recorded-weight and selected-case causal analysis

Saved evaluator judgments are trusted by default. The readout CLI replays them
only with the explicit `--audit-evaluations` option. Human B4–B6 labels remain in
separate annotation files; neither these commands nor an LLM judge supply them.

`condition_comparisons.csv` and `alignment_comparisons.csv` now include exact
McNemar p-values and conservative 95% paired accuracy-difference intervals.
Clopper–Pearson intervals bound both discordant cell probabilities with a
Bonferroni correction, then subtract their endpoints. Each step uses paired
**items**, never pooled steps. Coverage is pointwise and does not correct for
searching many steps/conditions. With two selected cases per benchmark these are
exploratory uncertainty summaries, not population performance or significance.

```bash
CUDA_VISIBLE_DEVICES=2 just analyze-latent-weights storage/experiments/latent_trajectory/arc_challenge/Qwen3-14B/d61ae9e14000e1c8
CUDA_VISIBLE_DEVICES=2 just analyze-latent-causal storage/experiments/latent_trajectory/arc_challenge/Qwen3-14B/d61ae9e14000e1c8 --steps 80 --conditions c d --generate
```

Both commands accept multiple source directories. Recorded revision, tokenizer
and dtype are used; weights must already be local. The weight command loads
only input embeddings and LM head. It projects saved actual-start hidden states
(C/D **after** closing prefill) and restores injection step k from hidden k-1,
the recorded alignment matrix and target norm. The alignment digest must match.
Step 0 raw prompt-state logits are available independently of readout completion
in `weight_analysis/trajectory_origin_logits.csv`. Readout and injection outputs
are `weight_analysis/readout_start_logits.csv` and
`weight_analysis/injected_vectors.csv` under the collection's analysis directory.
A missing readout remains missing; no generation is performed.

The causal command loads the frozen model once and analyzes selected saved
readouts (default steps 0/40/80, C/D). Conditional scoring forces an explicit
answer prefix, then reports first-token, whole-string and per-token log
probability separately. It excludes prefix likelihood and does not append EOS.
Choice candidates are A–D. GSM8K uses the gold numeral and saved predictions;
MBPP uses saved evaluator-extracted programs, never test assertions. Non-choice
tasks use up to two shortest distinct candidates per evaluator class. Missing
correct/incorrect alternatives leave the margin unavailable. Candidate string
scores are not semantic answer probabilities and depend on forced context.

E5 teacher-forces the closing token (C/D) and the first eight saved generated
tokens. Attention replay uses the eager backend and recorded tokens; generation
and conditional scoring use SDPA. Eager replay can differ numerically from the
original SDPA computation. Each query/head/layer reports attention mass over
prompt, latent and readout positions. A/B first-token prediction has no saved query; its replay
starts at the first generated token, which predicts the next token. Attention
mass does not establish useful information.

F2–F5 run only C/D, whose common closing-token query recomputes the start state
after KV intervention. Controls are original, exact self-transplant, persistent
latent attention blocking, zero latent V, latent position shuffle and other-item
latent KV transplant. `--region START END` restricts the half-open latent region.
Physical cache length, position numbering, prompt KV, closing query, seed,
sampling settings and original generation budget are fixed. Donor/shuffled K
are rephased to destination RoPE positions; donor V preserves its original norm.
The implementation supports full-attention Qwen3 dense KV. It creates one needed
device prefix per branch, with no full source clone or duplicated KV artifacts.

`--generate` additionally evaluates newly generated continuations with the
existing evaluator. Original baselines are rerun under the same protocol and
compared to self-transplant and perturbation controls. F6 records correctness
and conditional candidate-margin deltas per case. A changed outcome can show
latent-state sensitivity; it does not by itself establish newly computed or
progressively accumulated reasoning. The same seed is not full sampling control.
F1 uses existing stepwise readouts and is not recollected.

Causal outputs are under `causal_analysis/PROTOCOL_HASH/`, with one result JSON
and attention CSV per sample/condition/step, plus coverage and intervention
tables. Region, query count and generation mode identify the protocol directory.
Existing result JSONs are reused as authoritative without freshness checks or
automatic reruns. Use a different explicit protocol for a different experiment.
