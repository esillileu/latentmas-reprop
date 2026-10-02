import copy

from latentmas_reprop.application.receiver_reasoning.analysis import analyze_records
from latentmas_reprop.application.receiver_reasoning.report import export_analysis


def test_no_handoff_cells_and_gains(pilot, tmp_path):
    rows, config, metadata = pilot
    config["include_no_handoff"] = True
    for row in rows:
        row["config"] = copy.deepcopy(config)
    for row in list(rows):
        if row["upstream_latent_steps"] == 10:
            baseline = copy.deepcopy(row)
            baseline.update(
                upstream_latent_steps=0,
                context_id=f"{row['sample_id']}:0",
                upstream_context_sequence_length=0,
                correct=False,
            )
            rows.append(baseline)
    results = analyze_records(rows, config, metadata, bootstrap_count=100)
    assert not results[0]["integrity"]["issues"]
    assert len(results[0]["cells"]) == 6
    assert results[0]["derived"]["Handoff_gain_low_free"]["estimate"] == 0.75
    export_analysis(tmp_path, *results, rows)
    assert "no_handoff_answer_only" in (tmp_path / "summary.md").read_text()


def test_fixed_width_integrity_rejects_mismatched_handoff(pilot):
    rows, config, metadata = pilot
    config["handoff_positions"] = 10
    for row in rows:
        row["config"] = copy.deepcopy(config)
        row["handoff_positions"] = 10
        row["upstream_context_sequence_length"] = 10
    rows[0]["handoff_positions"] = 40
    metrics = analyze_records(rows, config, metadata, bootstrap_count=10)[0]
    assert metrics["derived"]["D_low"]["estimate"] is None
    assert metrics["derived"]["D_high"]["estimate"] is not None


def test_single_upstream_level_has_no_compute_contrasts(pilot, tmp_path):
    rows, config, metadata = pilot
    rows = [r for r in rows if r["upstream_latent_steps"] == 10]
    config["upstream_steps"] = [5]
    config["include_no_handoff"] = True
    config["handoff_positions"] = 10
    for row in rows:
        row.update(
            upstream_latent_steps=5,
            context_id=f"{row['sample_id']}:5",
            handoff_positions=10,
            upstream_context_sequence_length=10,
            config=copy.deepcopy(config),
        )
    for row in list(rows):
        baseline = copy.deepcopy(row)
        baseline.update(
            upstream_latent_steps=0,
            context_id=f"{row['sample_id']}:0",
            handoff_positions=0,
            upstream_context_sequence_length=0,
        )
        rows.append(baseline)
    metadata["params"] = {k: str(v) for k, v in config.items()}
    results = analyze_records(rows, config, metadata, bootstrap_count=10)
    assert not results[0]["integrity"]["issues"]
    assert len(results[0]["cells"]) == 4
    assert set(results[0]["derived"]) == {
        "D_low",
        "D_no_handoff",
        "Handoff_gain_low_answer_only",
        "Handoff_gain_low_free",
    }
    export_analysis(tmp_path, *results, rows)
    assert "Single upstream level" in (tmp_path / "summary.md").read_text()


def test_full_cache_integrity_uses_source_lengths(pilot):
    rows, config, metadata = pilot
    config["handoff_positions"] = None
    config["handoff_mode"] = "full"
    for row in rows:
        row["config"] = copy.deepcopy(config)
        row["upstream_source_sequence_length"] = row["upstream_context_sequence_length"]
        row["handoff_positions"] = row["upstream_source_sequence_length"]
    metadata["params"] = {k: str(v) for k, v in config.items()}
    metrics = analyze_records(rows, config, metadata, bootstrap_count=10)[0]
    assert not metrics["integrity"]["issues"]
    rows[0]["handoff_positions"] = 10
    metrics = analyze_records(rows, config, metadata, bootstrap_count=10)[0]
    assert metrics["derived"]["D_low"]["estimate"] is None
