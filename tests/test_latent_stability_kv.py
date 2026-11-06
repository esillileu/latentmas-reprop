"""Observable repetition and script diagnostics without a semantic judge."""

import json

import pytest

from latentmas_reprop.application.latent_output_stability import output_stability


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
