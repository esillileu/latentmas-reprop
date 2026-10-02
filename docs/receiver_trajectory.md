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
# Parity only on ten canonical representatives (one per digit)
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

Each model directory under `artifacts/receiver_trajectory/` contains:

- `sample_results.jsonl`: 2100 complete `ReceiverAcquisitionRecord` observations
  (100 drop + 2000 own). Includes sample/source IDs and digits; ten candidate
  probabilities/log-probabilities; argmax, candidate mass, rank, margin and
  source/target scores; top-five vocabulary tokens; KV length/bytes/layers/dtype
  and original/retained positions; receiver prompt length; latency and errors.
- `sender_latent_states.pt`: the existing probe collection protocol, with 20
  prompt templates × ten digits × 20 steps. Stores `hidden_pre_realign`,
  `latent_post_realign`, digit labels, template groups and example metadata.
  These 200 examples are distinct from the balanced 100 receiver samples.
- `candidate_token_mapping.json`: token IDs, contextual candidate mapping and
  receiver prompt hash from the acquisition scoring helper.
- `receiver_input.json`: actual receiver input token IDs and attention mask.
- `resolved_config.json`: the collection command options and input/output paths.
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
metadata. Root outputs retain all 21 observations without overwriting steps.
Target-digit expectations, source-follow feedback and execution status follow
the existing acquisition tracker. Traces are flushed before the run ends.
Runs completed before this tracing fix have artifacts but no sample traces;
real forward traces require a new execution.

Previously completed 0.6B artifacts predate sender-state collection; that run
contains receiver observations and probe-joined summaries, but not the newly
added sender-state artifact. No historical run is modified or backfilled.

## Offline analysis

Fit probes later using the existing `just analyze-sender-probe` command and the
saved sender states. Once model-specific probe cells are exported, join them to
saved receiver records without loading a language model:

```bash
just analyze-receiver-trajectory \
  --probe-cells artifacts/receiver_acquisition/sender_probe_cells.csv
```

The analysis reads complete collected model directories, calculates paired
argmax change fractions, and matches each step to the full20 run's
`latent_post_realign` probe. It writes `analysis/trajectory.csv`, `.json` and
PNG/PDF scatter under `artifacts/receiver_trajectory/`. All four models produce
80 rows; the first three produce 60 rows; a single model produces 20 rows.
Missing probe cells fail explicitly. Analysis does not upload to MLflow.

Columns include `model`, `latent_step`, `probe_accuracy`, `probe_null_mean`,
`probe_effect_pp`, `probe_fwer_p`, and `receiver_changed_fraction`, plus sample
count and probe provenance. X is probe accuracy minus null mean in pp; Y is
receiver changed fraction × 100. Models have fixed colors and thin step-order
connections, without regression, correlation or per-point labels.
`just plot-receiver-trajectory --input <analysis CSV>` can redraw the scatter.
Paths can be overridden with the commands' input/output arguments.
