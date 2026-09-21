from latentmas_reprop.infrastructure.evaluators import (
    extract_gold,
    extract_gsm8k_answer,
    extract_markdown_python_block,
    get_evaluator,
    normalize_answer,
)


def test_extract_gsm8k_answer_boxed():
    text = "The answer is therefore \\boxed{42}."
    assert extract_gsm8k_answer(text) == "42"


def test_extract_gsm8k_answer_number_fallback():
    text = "Total eggs are 18"
    assert extract_gsm8k_answer(text) == "18"


def test_extract_gold():
    solution = "She sells 9 * 2 = 18 eggs\n#### 18"
    assert extract_gold(solution) == "18"


def test_normalize_answer():
    assert normalize_answer("  42  ") == "42"
    assert normalize_answer("A") == "a"
    assert normalize_answer(None) is None


def test_extract_markdown_python_block():
    text = """Here is the solution:
```python
def add(a, b):
    return a + b
```
Hope this helps!"""
    code = extract_markdown_python_block(text)
    assert code is not None
    assert "def add(a, b):" in code


def test_evaluator_evaluate_gsm8k():
    evaluator = get_evaluator()
    pred, ok, err = evaluator.evaluate(
        task="gsm8k",
        raw_prediction="Hence \\boxed{18}",
        gold="18",
    )
    assert pred == "18"
    assert ok is True
    assert err is None
