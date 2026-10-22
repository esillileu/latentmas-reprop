# Presentation plots

```bash
just plot-presentation
```

The entry point `src/run/presentation_plots.py` uses saved analysis CSVs and
the supplied nine secret-digit scatter coordinates.
It does not load models, run inference, recompute statistics, or write to MLflow.
Missing or duplicate cells fail rather than pooling runs. Dependencies and lockfiles
are unchanged. Existing analysis and diagnostic plotting remain unchanged.

Required local inputs:

- `artifacts/receiver_acquisition/sender_probe_cells.csv`
- `artifacts/receiver_acquisition/receiver_conditions.csv`
- `artifacts/receiver_compute_preflight/Qwen_Qwen3-4B/budget_curves.csv`
- `artifacts/receiver_compute_preflight/Qwen_Qwen3-14B/budget_curves.csv`
- Each budget directory's `source.json`, identifying the corresponding model.

The acquisition inputs are the existing `just analyze-receiver` outputs. Budget
inputs follow the layout of `just analyze-receiver-compute-preflight --model 4B`
and `--model 14B`. For this presentation they were downloaded directly from
existing MLflow `analysis/` artifacts without rerunning analysis or uploading:

| Model | Source run ID |
| --- | --- |
| Qwen/Qwen3-4B | `8c3949d5499842b6a6052183c7ae769c` |
| Qwen/Qwen3-14B | `ff70508e50d34bc3af4165bd99be63a4` |

Outputs under `artifacts/presentation/`, each as PNG and PDF:

- `secret_digit_probe`: Qwen3-8B, 20 latent steps; pre/post realignment probe
  accuracy. Filled/hollow markers use the saved FWER-corrected significance flag.
  The only reference is 10% chance.
- `secret_digit_receiver`: the same model/steps, with drop, latent-only own,
  and latent-only cross mapped to No Handoff, Matched, and Mismatched. The
  accuracy axis starts at zero and ends at 25%; chance is 10%.
- `receiver_budget_4b_14b`: identical 4B/14B panels at 20 upstream steps,
  budgets 64/128/256/512/1024 on a base-2 logarithmic axis, and matched,
  mismatched, no-handoff lines and markers, without accuracy confidence intervals.
  Both accuracy axes span 0–100%; the only reference is 50%.
- `secret_digit_aspects`: nine supplied model × step coordinates comparing
  final post-realignment probe OOF accuracy minus permutation null mean (pp)
  against receiver argmax prediction changes for latent-only own versus matched
  drop (%). Each model's steps 1 → 4 → 20 are connected in order, with only step
  labels beside the points. Hollow 4B step 4/20 markers denote probes that are
  not FWER-significant. These rounded coordinates are used exactly as supplied,
  without recomputing metrics; no regression or correlation is shown.

Override directories with `--acquisition-dir`, `--compute-dir`, or `--output-dir`.
The repository ignores `artifacts/`; generated figures and downloaded inputs
remain local files, while the plotting entry point and instructions are tracked.
