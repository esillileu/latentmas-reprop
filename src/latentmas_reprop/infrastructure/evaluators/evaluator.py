import re
import traceback
from multiprocessing import Manager, Process

from ...domain.ports.evaluator_port import EvaluatorPort


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract final answer from \\boxed{...} or fallback to numbers."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()

    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


def extract_gold(text: str) -> str | None:
    """Extract gold answer from GSM8K text format (#### <answer>)."""
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    return match.group(1) if match else None


def normalize_answer(ans: str | None) -> str | None:
    """Normalize answer string by stripping whitespace and lowercasing."""
    if ans is None:
        return None
    return ans.strip().lower()


def extract_markdown_python_block(text: str) -> str | None:
    """Extract python code inside ```python ... ``` markdown code block."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def run_with_timeout(code: str, timeout: int = 10) -> tuple[bool, str | None]:
    """Execute python code in a separate process with a timeout."""

    def worker(ns, c):
        try:
            local_ns = {}
            exec(c, local_ns)
            ns["ok"] = True
            ns["error"] = None
        except Exception:
            ns["ok"] = False
            ns["error"] = traceback.format_exc()

    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            ns["ok"] = False
            ns["error"] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return ns.get("ok", False), ns.get("error", None)


class StandardEvaluator(EvaluatorPort):
    """Adapter implementing EvaluatorPort for various benchmark datasets."""

    def evaluate(
        self,
        task: str,
        raw_prediction: str,
        gold: str | None,
    ) -> tuple[str | None, bool, str | None]:
        if task in ["mbppplus", "humanevalplus"]:
            pred = extract_markdown_python_block(raw_prediction)
            if pred is None:
                return None, False, "python error: No python code block found"
            python_code_to_exe = pred + "\n" + (gold or "")
            ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
            return pred, ok, error_msg

        if task in ["aime2024", "aime2025"]:
            pred = normalize_answer(extract_gsm8k_answer(raw_prediction))
            gold_str = str(gold).strip() if gold is not None else ""
            try:
                pred_int = int(pred) if pred is not None else None
                gold_int = int(gold_str)
                ok = pred_int == gold_int
                error_msg = None
            except (ValueError, TypeError):
                ok = False
                error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            return pred, ok, error_msg

        # GSM8K, ARC, GPQA, MedQA, Winogrande, etc.
        pred = normalize_answer(extract_gsm8k_answer(raw_prediction))
        gold_norm = normalize_answer(gold) if gold is not None else None
        ok = (
            (pred == gold_norm)
            if (pred is not None and gold_norm is not None)
            else False
        )
        return pred, ok, None


DEFAULT_EVALUATOR = StandardEvaluator()


def get_evaluator() -> StandardEvaluator:
    return DEFAULT_EVALUATOR
