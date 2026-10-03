"""Strict offline joins, parity failure artifacts and unsmoothed figure inputs."""

import csv

import pytest

from src.run import stepwise_communication_analysis as analysis


def inputs():
    probes, sweep, independent = [], [], []
    for model in analysis.MODELS[:3]:
        for step in range(1, 21):
            probes.append(
                {
                    "model": model,
                    "latent_steps": "20",
                    "step": str(step),
                    "representation": "latent_post_realign",
                    "probe_accuracy": "0.2",
                    "null_mean_accuracy": "0.11",
                    "fwer_p_value": "0.05",
                }
            )
            sweep.append(
                {
                    "model": model,
                    "latent_step": str(step),
                    "receiver_changed_fraction": str(step / 20),
                }
            )
            if step in (1, 4, 20):
                independent.append(
                    {
                        "model": model,
                        "latent_steps": str(step),
                        "condition": "latent_only/own",
                        "run_id": "independent",
                        "argmax_changed_fraction": str(step / 20),
                    }
                )
    return probes, sweep, independent


def test_join_derives_metrics_and_uses_strict_fwer_threshold():
    probes, sweep, independent = inputs()
    probes[1]["fwer_p_value"] = "0.049"
    rows = analysis.communication_metrics(probes, list(reversed(sweep)))
    assert len(rows) == 60
    assert list(rows[0]) == list(analysis.COLUMNS)
    assert rows[0]["probe_effect_pp"] == pytest.approx(9)
    assert rows[0]["receiver_changed_pct"] == 5
    assert rows[0]["probe_significant"] is False
    assert rows[1]["probe_significant"] is True
    assert len(analysis.receiver_parity(rows, independent)) == 9


@pytest.mark.parametrize("source", (0, 1))
@pytest.mark.parametrize(
    "corruption", ("missing", "duplicate", "nan", "empty", "field")
)
def test_invalid_inputs_fail_without_dropping_rows(source, corruption):
    data = inputs()
    rows = data[source]
    field = "probe_accuracy" if source == 0 else "receiver_changed_fraction"
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0].copy()
    elif corruption == "field":
        del rows[0][field]
    else:
        rows[0][field] = "nan" if corruption == "nan" else ""
    with pytest.raises(ValueError):
        analysis.communication_metrics(*data[:2])


@pytest.mark.parametrize("corruption", ("missing", "duplicate", "nan"))
def test_parity_requires_all_nine_valid_independent_cells(corruption):
    probes, sweep, independent = inputs()
    if corruption == "missing":
        independent.pop()
    elif corruption == "duplicate":
        independent[-1] = independent[0].copy()
    else:
        independent[0]["argmax_changed_fraction"] = "nan"
    with pytest.raises(ValueError):
        analysis.receiver_parity(
            analysis.communication_metrics(probes, sweep), independent
        )


def test_parity_failure_saves_table_and_prevents_plot(tmp_path, monkeypatch):
    data = inputs()
    data[2][0]["argmax_changed_fraction"] = "0.9"
    monkeypatch.setattr(
        analysis, "plot_metrics", lambda *args: pytest.fail("Unexpected plot")
    )
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="parity failed"):
        analysis.analyze(*data, output)
    assert (output / "stepwise_receiver_parity.csv").exists()


def test_artifacts_use_exact_final_rows_and_plot_has_no_default_annotations(
    tmp_path, monkeypatch
):
    figures = []
    monkeypatch.setattr(analysis, "save", lambda fig, *args: figures.append(fig))
    rows = analysis.analyze(*inputs(), tmp_path / "output")
    with (tmp_path / "output/stepwise_communication_metrics.csv").open() as stream:
        saved = list(csv.DictReader(stream))
    assert len(saved) == 60
    assert [float(row["receiver_changed_pct"]) for row in saved] == [
        row["receiver_changed_pct"] for row in rows
    ]
    ax = figures[0].axes[0]
    assert not ax.texts
    assert ax.get_xlabel() == "Digit signal in handoff latent (pp above null)"
    assert len([line for line in ax.lines if len(line.get_xdata()) == 20]) == 3
    assert len(figures[1].axes) == 6
    diagnostic_axes = figures[1].axes
    assert all(axis.get_xlim() == (0.5, 20.5) for axis in diagnostic_axes)
    assert all(axis.get_ylim() == ax.get_xlim() for axis in diagnostic_axes[:3])
    assert all(axis.get_ylim() == ax.get_ylim() for axis in diagnostic_axes[3:])
    assert all(
        list(axis.get_yticks()) == list(ax.get_xticks()) for axis in diagnostic_axes[:3]
    )
    assert all(
        list(axis.get_yticks()) == list(ax.get_yticks()) for axis in diagnostic_axes[3:]
    )
    for fig in figures:
        analysis.plt.close(fig)


def test_four_models_include_explicit_unavailable_14b_parity():
    probes, sweep, independent = inputs()
    model = analysis.MODELS[-1]
    probes.extend(row | {"model": model} for row in list(probes[:20]))
    sweep.extend(row | {"model": model} for row in list(sweep[:20]))
    metrics = analysis.communication_metrics(probes, sweep)
    parity = analysis.receiver_parity(metrics, independent)
    assert len(metrics) == 80 and len(parity) == 12
    assert sum(row["matches"] is True for row in parity) == 9
    missing = [row for row in parity if row["model"] == model]
    assert len(missing) == 3
    assert all(
        row["status"] == "not_available" and row["matches"] is None for row in missing
    )
