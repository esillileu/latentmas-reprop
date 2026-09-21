from collections.abc import Iterable
from typing import Any

from datasets import load_dataset

from ..evaluators.evaluator import extract_gold, normalize_answer
from ..paths.resolver import PathResolver, get_path_resolver


def load_gsm8k(split: str = "test", cache_dir: str | None = None) -> Iterable[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: str | None = None) -> Iterable[dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(split: str = "train", cache_dir: str | None = None) -> Iterable[dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(
    split: str = "test", cache_dir: str | None = None
) -> Iterable[dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def _map_arc_label(label: Any) -> str:
    label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}
    s = str(label).strip()
    return label_map.get(s, s.lower())


def load_arc_easy(split: str = "test", cache_dir: str | None = None) -> Iterable[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts, strict=False):
            mlabel = _map_arc_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = _map_arc_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(
    split: str = "test", cache_dir: str | None = None
) -> Iterable[dict]:
    ds = load_dataset(
        "allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir
    )
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts, strict=False):
            mlabel = _map_arc_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = _map_arc_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: str | None = None,
) -> Iterable[dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = "Pickout proper choice that fits the _ in the following sentence:"
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str | None = None,
    cache_dir: str | None = None,
) -> Iterable[dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""
        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str | None = None,
    cache_dir: str | None = None,
) -> Iterable[dict]:
    ds = load_dataset(
        "evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir
    )
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace("candidate", item["entry_point"])
        answer += f"\n\ncheck({item['entry_point']})"
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_medqa(
    split: str | None = None,
    subset: str | None = None,
    cache_dir: str | None = None,
    path_resolver: PathResolver | None = None,
) -> Iterable[dict]:
    resolver = path_resolver or get_path_resolver()
    medqa_file = str(resolver.medqa_json_path)

    ds = load_dataset("json", data_files=medqa_file, split="train", cache_dir=cache_dir)
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
        answer = ""
        for idx, op in enumerate(item["options"]):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }
