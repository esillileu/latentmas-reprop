"""Prompt builder for baseline single-agent reasoning."""

from typing import Any


def build_agent_messages_single_agent(
    question: str, args: Any = None
) -> list[dict[str, str]]:
    system_message = (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

    assert args.method in ["baseline"], (
        "this prompt only for baseline method (single agent)"
    )
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    task = getattr(args, "task", "gsm8k")

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
