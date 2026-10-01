import copy

import pytest


@pytest.fixture
def pilot():
    config = {
        "model_name": "Qwen/test",
        "task": "gsm8k",
        "split": "test",
        "prompt": "sequential",
        "seed": 42,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 2048,
        "answer_only_max_new_tokens": 64,
        "upstream_steps": [10, 40],
    }
    patterns = [
        (False, True, True, True),
        (True, True, False, True),
        (False, False, False, True),
        (False, True, False, False),
    ]
    records = []
    for i, pattern in enumerate(patterns):
        for (step, mode), correct in zip(
            [(10, "answer_only"), (10, "free"), (40, "answer_only"), (40, "free")],
            pattern,
            strict=True,
        ):
            records.append(
                {
                    "sample_id": str(i),
                    "sample_index": i,
                    "question": f"q{i}",
                    "gold": "1",
                    "upstream_latent_steps": step,
                    "receiver_mode": mode,
                    "prediction": "1" if correct else "0",
                    "raw_receiver_output": "answer",
                    "correct": correct,
                    "receiver_generated_tokens": 5,
                    "receiver_token_limit_reached": False,
                    "context_id": f"{i}:{step}",
                    "upstream_context_sequence_length": 100 + step,
                    "model": config["model_name"],
                    "seed": 42,
                    "config": copy.deepcopy(config),
                }
            )
    metadata = {
        "run_id": "run",
        "experiment_id": "1",
        "status": "FINISHED",
        "start_time": 0,
        "end_time": 1,
        "artifact_uri": "uri",
        "params": {k: str(v) for k, v in config.items()},
        "metrics": {},
        "tags": {},
    }
    return records, config, metadata
