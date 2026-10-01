"""Receiver final-answer prompts, independent of upstream agent roles."""


def build_answer_only_messages(items: list[dict], task: str) -> list[list[dict]]:
    answer_format = (
        "a self-contained Python function in a markdown Python code block"
        if task in {"mbppplus", "humanevalplus"}
        else r"\boxed{YOUR_FINAL_ANSWER}"
    )
    return [
        [
            {
                "role": "system",
                "content": "Return only the final answer. Do not output reasoning, plans, feedback, explanations, or a thinking trace.",
            },
            {
                "role": "user",
                "content": (
                    f"Target Question: {item['question']}\n"
                    "Use the preceding latent information as reference when helpful. "
                    f"Output only {answer_format}. /no_think"
                ),
            },
        ]
        for item in items
    ]
