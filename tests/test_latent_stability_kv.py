"""Observable output diagnostics and nonmutating saved KV structure analysis."""

import json

import pytest
import torch

from latentmas_reprop.application.latent_kv_analysis import analyze_kv
from latentmas_reprop.application.latent_kv_features import kv_component_features
from latentmas_reprop.application.latent_output_stability import output_stability
from src.run.latent_kv_analysis import write_kv_report


def record(text, ids=None):
    content = ids if ids is not None else list(map(ord, text))
    return {
        "generated_text": text,
        "generated_token_ids": [*content, 999],
        "content_length": len(content),
    }


def test_repetition_excludes_eos_and_does_not_classify_scripts_as_languages():
    result = output_stability(
        record("same line\nsame line\nabc \u03b1 def", [1, 2] * 8)
    )
    assert result["repeated_4gram_fraction"] > 0.5
    assert result["longest_identical_token_run"] == 1
    assert result["repeated_line_fraction"] == pytest.approx(1 / 3)
    assert result["script_count"] == 2
    assert result["script_switch_count"] == 2
    assert json.loads(result["script_counts"])["greek"] == 1
    tags = output_stability(record("<think>中文</think>"))
    assert json.loads(tags["script_counts"]) == {"han": 2}
    assert tags["script_switch_count"] == 0
    assert (
        output_stability(record("<|im_start|>assistant"))["generated_chat_open_count"]
        == 1
    )
    assert output_stability(record("```python\nprint(1)"))["odd_code_fence_count"]
    assert not output_stability(record("```python\nprint(1)\n```"))[
        "odd_code_fence_count"
    ]


def test_empty_diagnostics_have_missing_ratios_not_false_zero_scores():
    result = output_stability(record(""))
    assert result["repeated_4gram_fraction"] is None
    assert result["repeated_line_fraction"] is None
    assert result["non_latin_letter_fraction"] is None
    assert result["longest_identical_token_run"] == 0
    assert output_stability(record("x", [1] * 10))["longest_identical_token_run"] == 10


def tensor():
    return torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]]])


def test_kv_position_mapping_similarity_rank_and_nonmutation():
    values = tensor()
    before = values.clone()
    summary, rows = kv_component_features(values, 2, 2)
    assert torch.equal(values, before)
    assert [r["cache_position"] for r in rows] == [0, 1, 2, 3]
    assert [r["region"] for r in rows] == ["prompt", "prompt", "latent", "latent"]
    rows = {r["step"]: r for r in rows if r["step"] is not None}
    assert rows[0]["adjacent_cosine"] == 0
    assert rows[1]["adjacent_cosine"] == 0
    assert rows[2]["adjacent_cosine"] == 1
    assert rows[1]["nearest_prompt_position"] == 0
    assert rows[1]["nearest_prompt_cosine"] == 1
    assert rows[1]["prompt_centroid_cosine"] == pytest.approx(2**-0.5)
    assert summary["latent_effective_rank"] == pytest.approx(1)
    assert summary["latent_numerical_rank"] == 1
    assert summary["latent_top1_energy_fraction"] == pytest.approx(1)
    assert summary["latent_std"] == 0.5
    assert not summary["rank_centered"]
    with pytest.raises(ValueError, match="shape"):
        kv_component_features(values, 2, 3)


def test_nonfinite_positions_do_not_bridge_and_zero_cosine_is_unavailable():
    values = tensor()
    values[..., 2, 0] = float("nan")
    summary, rows = kv_component_features(values, 2, 2)
    assert summary["status"] == "partial"
    assert summary["valid_latent_positions"] == 1
    assert rows[2]["status"] == "invalid" and rows[2]["norm"] is None
    assert rows[3]["adjacent_cosine"] is None
    summary, rows = kv_component_features(torch.zeros_like(values), 2, 2)
    assert summary["latent_effective_rank"] == 0
    assert rows[1]["adjacent_cosine"] is None
    assert rows[1]["nearest_prompt_cosine"] is None


def test_saved_cache_analysis_is_read_only_and_handles_missing_layers(tmp_path):
    root = tmp_path / "run"
    sample = root / "samples/000000"
    cache_dir = sample / "trajectories/without_think"
    cache_dir.mkdir(parents=True)
    (root / "metadata.json").write_text(
        json.dumps(
            {"task": "gsm8k", "model_name": "model", "latent_space_realign": False}
        )
    )
    (sample / "sample.json").write_text(json.dumps({"index": 0}))
    path = cache_dir / "kv.pt"
    torch.save(
        {
            "past_key_values": ((tensor(), tensor()),),
            "prompt_length": 2,
            "latent_steps": 2,
        },
        path,
    )
    original = path.read_bytes()
    statuses, summaries, steps = analyze_kv(root, layers=[0, 99])
    assert path.read_bytes() == original
    assert [r["status"] for r in statuses] == ["partial", "missing"]
    assert len(steps) == 8 and len(summaries) == 3
    assert summaries[-1]["status"] == "invalid"
    output = tmp_path / "report"
    write_kv_report(statuses, summaries, steps, output)
    assert (output / "kv_positions.csv").exists()
    assert (output / "kv_rank_000000_without_think.png").exists()
    assert json.loads((output / "kv_status.json").read_text())[1]["status"] == "missing"
    assert analyze_kv(root, samples=[145]) == ([], [], [])


def test_prompt_spectrum_and_similarity_use_all_positions_without_self_matches():
    summary, rows = kv_component_features(tensor(), 2, 2)
    assert summary["prompt_effective_rank"] == pytest.approx(2)
    assert summary["prompt_numerical_rank"] == 2
    assert summary["prompt_top1_energy_fraction"] == pytest.approx(0.5)
    assert summary["latent_effective_rank"] == pytest.approx(1)
    assert rows[0]["step"] is None and rows[1]["step"] == 0
    assert rows[0]["nearest_prompt_position"] == 1
    assert rows[0]["nearest_prompt_cosine"] == 0
    assert rows[1]["nearest_prompt_position"] == 0
    assert rows[1]["nearest_prompt_cosine"] == 0


@pytest.mark.parametrize(
    "prompt,latent,expected_prompt,expected_latent",
    [
        (
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0]],
            1,
            2,
        ),
        (
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            2,
            1,
        ),
    ],
)
def test_equal_length_rank_controls_for_region_length(
    prompt, latent, expected_prompt, expected_latent
):
    values = torch.tensor(prompt + latent)
    original = values.clone()
    summary, rows = kv_component_features(values, len(prompt), len(latent))
    assert torch.equal(values, original)
    assert summary["matched_window_positions"] == 2
    assert summary["matched_prompt_effective_rank"] == pytest.approx(expected_prompt)
    assert summary["matched_latent_effective_rank"] == pytest.approx(expected_latent)
    assert summary["matched_prompt_rank_positions"] == 2
    assert summary["matched_latent_rank_positions"] == 2
    assert rows[len(prompt)]["region"] == "latent"
    assert rows[len(prompt)]["step"] == 1
    assert rows[len(prompt)]["cache_position"] == len(prompt)
