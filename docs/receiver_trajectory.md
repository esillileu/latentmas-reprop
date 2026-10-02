# Secret-digit receiver trajectory

This experiment uses the existing receiver acquisition protocol: the canonical
100 balanced samples at seed 42, sender prompt v1, receiver prompt v2 with
thinking disabled, `The number is ` scoring prefix, contextual single-token
digit candidates, digit argmax, latent-only own handoff, and no-cache drop at
receiver position zero. It does not inject saved probe vectors into the receiver
and does not reuse the probe's 200 template/digit examples as receiver samples.

Each canonical receiver sample gets one new 20-step sender rollout. For step k,
copy its full KV cache, retain the first `prompt_len + k` positions, then retain
the last k positions. Set both `original_full_seq_len` and receiver
`position_start` to `prompt_len + k`. Original RoPE keys are retained. Never take
the last k positions directly from full20, which would use future states.
Evaluate k=1..20 and one shared drop for each sample.

## Parity gate

Before trajectory collection, ten representatives (one per digit, preserving
their original sample IDs/keys from the balanced 100) are checked at k=1 and 4:

- All layer key/value tensors must equal a fresh independent k-step rollout
  exactly, with `torch.equal`.
- Receiver next-token logits must equal exactly for both cache constructions.
- Candidate digit log-probabilities and predictions for own/drop, plus sample
  identities and receiver positions, must equal the saved canonical k-step runs.

Historical runs did not save raw KV. Historical cache equality cannot be claimed;
the exact cache comparison is against freshly generated independent rollouts.
`parity.json` records every check and historical log-probability differences.
There is no tolerance override or option to bypass a failed check. Failure stops
before writing trajectory records. Full model-specific execution additionally
requires successful 0.6B parity on the same GPU name, Git commit and library
versions. BF16 is selected explicitly without changing other commands' FP16 defaults.

## Local verification

```bash
just run-receiver-trajectory --smoke
```

Local smoke is restricted to 0.6B. If parity passes, it evaluates the 20 steps on
ten balanced representatives, saves raw records and a 20-row smoke CSV under
`artifacts/receiver_trajectory/smoke/`. This is not the final 100-sample experiment.
The presentation plot refuses smoke results.

The initial local check on an RTX 4060 Laptop GPU passed all 20 exact cache and
fresh receiver-logit comparisons. Historical argmax predictions also matched
20/20, but historical digit log-probabilities did not match exactly (maximum
absolute difference 1.102187). Accordingly, the parity gate refused trajectory
collection. This documents an unresolved historical numerical reproducibility
difference; it does not establish its cause. No precision, package, lockfile or
environment changes were made to force a match.

## Full execution on a 32GB GPU

Prepare the same checkout, existing dependencies, source model weights, and
the historical receiver environment. Copy these existing analysis inputs:

- `artifacts/receiver_acquisition/sender_probe_cells.csv`
- `artifacts/receiver_acquisition/receiver_conditions.csv`
- Historical `sample_results.jsonl` for each model's canonical 1/4-step runs,
  at `.cache/receiver_acquisition/mlflow/<run_id>/sample_results.jsonl`.
  Run IDs are selected from `receiver_conditions.csv`, not guessed by recency.

```bash
just run-receiver-trajectory
just plot-receiver-trajectory
```

Full inference refuses devices with less than 30 GiB total VRAM and requires
native BF16. It processes 0.6B, 4B, then 8B sequentially, checking parity for each
model before its trajectory. Historical exact parity must first be resolved in
the intended execution environment; a 32GB GPU alone does not guarantee it.

Each model directory under `artifacts/receiver_trajectory/` contains provenance,
parity evidence, 2100 raw receiver observations (100 drop + 2000 own), and a
20-row `trajectory.csv`. The combined CSV is written only after all three models
finish. It contains exactly 60 cells, pairing each receiver change fraction to
the existing 20-step run's post-realignment probe effect:

- X: `100 * (probe OOF accuracy - permutation null mean)`, in pp.
- Y: `100 * mean(own predicted_digit != paired drop predicted_digit)`, in %.

`plot-receiver-trajectory` only reads the combined CSV and writes
`artifacts/presentation/secret_digit_trajectory.png` (300 dpi) and `.pdf`.
Models use consistent colors and thin step-order connections; hollow markers
denote saved FWER-nonsignificant probe cells. No regression, correlation, CI,
or per-point numeric labels are added. X spans all observed effects, including
negative values and values above 10 pp; Y remains 0–100%.

Paths can be overridden with `--acquisition-dir`, `--reference-dir`, and
`--output-dir` for inference, or `--input` and `--output-dir` for plotting.
These commands write only local artifacts and do not modify historical MLflow runs.
