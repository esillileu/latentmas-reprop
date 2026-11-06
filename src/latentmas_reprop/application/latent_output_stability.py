"""Observable repetition, script mixing and structural-marker diagnostics."""

import json
import re
import unicodedata
from collections import Counter
from itertools import groupby, pairwise


def character_script(character):
    if not unicodedata.category(character).startswith("L"):
        return None
    name = unicodedata.name(character, "")
    for label in ("LATIN", "HANGUL", "CYRILLIC", "ARABIC", "GREEK", "DEVANAGARI"):
        if label in name:
            return label.lower()
    if "CJK" in name or "IDEOGRAPH" in name:
        return "han"
    if "HIRAGANA" in name or "KATAKANA" in name:
        return "kana"
    return "other_letter"


def output_stability(record):
    """Counts are descriptive; script changes do not diagnose language or errors."""
    tokens = record["generated_token_ids"][: record["content_length"]]
    grams = Counter(tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3)))
    total = sum(grams.values())
    raw = record["generated_text"]
    text = re.sub(r"<\|[^<>]*\|>|</?think>", "", record.get("evaluation_text", raw))
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    repeated_lines = sum(count - 1 for count in Counter(lines).values())
    scripts = [
        script
        for character in text
        if (script := character_script(character)) is not None
    ]
    counts = Counter(scripts)
    return {
        "repeated_4gram_fraction": sum(count - 1 for count in grams.values()) / total
        if total
        else None,
        "longest_identical_token_run": max(
            (sum(1 for _ in group) for _, group in groupby(tokens)), default=0
        ),
        "nonempty_line_count": len(lines),
        "repeated_line_fraction": repeated_lines / len(lines) if lines else None,
        "script_counts": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
        "script_count": len(counts),
        "script_switch_count": sum(a != b for a, b in pairwise(scripts)),
        "non_latin_letter_fraction": (len(scripts) - counts["latin"]) / len(scripts)
        if scripts
        else None,
        "generated_chat_open_count": raw.count("<|im_start|>"),
        "odd_code_fence_count": len(re.findall(r"(?m)^\s*`{3,}", text)) % 2 == 1,
    }
