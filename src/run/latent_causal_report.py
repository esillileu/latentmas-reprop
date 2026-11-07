"""Tables and interpretation notes for selected latent causal cases."""

from pathlib import Path

from latentmas_reprop.infrastructure.storage import save_json_atomic

from .latent_trajectory_report import write_csv


def output_directory(directory, metadata):
    return (
        Path("artifacts/latent_trajectory")
        / metadata["task"]
        / directory.parent.name
        / directory.name
        / "causal_analysis"
    )


def write_summary(output, cases):
    table = [
        {
            **{
                k: case[k]
                for k in (
                    "source_run_id",
                    "benchmark",
                    "sample_id",
                    "alignment",
                    "condition",
                    "step",
                )
            },
            "case_status": case["status"],
            **{
                k: v for k, v in result.items() if k not in ("scores", "generated_text")
            },
        }
        for case in cases
        for result in case.get("results", [])
    ]
    write_csv(output / "interventions.csv", table)
    save_json_atomic(
        output / "coverage.json",
        [{k: v for k, v in case.items() if k != "results"} for case in cases],
    )
    (output / "summary.md").write_text(
        "# Selected-case causal analysis\n\nC5: teacher-forced candidates after the recorded forced output prefix. First-token, full-string and mean-token scores are distinct. These are probabilities of specific strings, not semantic answer probabilities. MBPP scores saved evaluator-extracted programs, never gold test assertions.\n\nE5: teacher-forced saved continuation queries (C/D include the closing token). Each layer/head/query reports prompt, latent and readout attention mass. A/B has no recorded query for the very first predicted token. Attention mass does not measure useful information.\n\nF2: persistent latent attention blocking with length/positions fixed. F3: zero latent V with original K. F4: other same-run item latent KV, with K rephased from donor to receiver RoPE positions. F5: exact self-transplant and position-rephased shuffled latent controls. Prompt KV, closing token, generation settings, budget and seed are fixed. No source KV is changed or saved again.\n\nF6: paired original/intervention correctness deltas and candidate-margin deltas are case evidence. The same seed does not establish sampling invariance. Check self-transplant and shuffled controls before interpreting changes. No population significance, newly computed reasoning, or absence of latent usefulness follows from these few selected cases.\n\nExisting JSON results are trusted and reused. Coverage is explicit; pending readouts are omitted from effects. Human meaning/progress/restart labels remain separate and unreviewed.\n\n[Interventions](interventions.csv) · [Coverage](coverage.json)\n"
    )
