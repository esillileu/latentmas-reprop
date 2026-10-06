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
- `artifacts/receiver_compute_preflight/Qwen_Qwen3-8B/budget_curves.csv`
- Each budget directory’s `metrics.json`, containing saved paired comparisons.
- Each budget directory's `source.json`, identifying the corresponding model.

The acquisition inputs are the existing `just analyze-receiver` outputs. Budget
inputs follow the layout of `just analyze-receiver-compute-preflight --model 4B`
and `--model 14B`. For this presentation they were downloaded directly from
existing MLflow `analysis/` artifacts without rerunning analysis or uploading:

| Model | Source run ID |
| --- | --- |
| Qwen/Qwen3-4B | `8c3949d5499842b6a6052183c7ae769c` |
| Qwen/Qwen3-14B | `ff70508e50d34bc3af4165bd99be63a4` |

Outputs under `artifacts/presentation/` (secret-digit figures: PNG/PDF; budget figures: PNG/SVG):

- `secret_digit_probe`: Qwen3-8B, 20 latent steps; pre/post realignment probe
  accuracy. Filled/hollow markers use the saved FWER-corrected significance flag.
  The only reference is 10% chance.
- `secret_digit_receiver`: the same model/steps, with drop, latent-only own,
  and latent-only cross mapped to No Handoff, Matched, and Mismatched. The
  accuracy axis starts at zero and ends at 25%; chance is 10%.
- Three model-specific solver budget figures and three content-effect figures,
  described below (SVG and PNG).
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

## Solver budget and content effect figures

```bash
just plot-presentation --budget-only
```

Reads `budget_curves.csv`, `metrics.json`, and `source.json` from each
`Qwen_Qwen3-{4B,8B,14B}` directory under `artifacts/receiver_compute_preflight/`.
All 30 finite condition cells and 10 saved `pairing_gain` comparisons per model
are required. Sample counts, model provenance, shared No latent baselines, finite
values, CI ordering, and agreement between effect estimates and accuracy differences
are checked. Free results are excluded before these checks; existing experiment,
condition, Free-generation, and verification tables are not rewritten.

For each model, `receiver_budget_{model}` and `receiver_content_effect_{model}`
are saved as SVG and 300-dpi PNG in `artifacts/presentation/`. Both have 10/20-step
panels with log₂ budget ticks 64/128/256/512/1024. Accuracy uses a shared 0–100%
scale and a single figure legend. Content effects use the saved paired bootstrap
95% CI from `metrics.json`, without recomputing statistics, selecting significant
points, or connecting effect estimates. All six effect panels share limits covering
all saved intervals and zero. Figures use white backgrounds, light grids, large
labels, and no point-value annotations. Directory overrides and `--sample-count`
remain available; the supplied sample count must match every model.
