"""Load the recorded trajectory tokenizer without model weights or downloads."""

import hashlib

from transformers import AutoTokenizer


def load_trajectory_tokenizer(metadata):
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["tokenizer"],
        revision=metadata.get("tokenizer_revision") or metadata.get("model_revision"),
        local_files_only=True,
    )
    identity = tokenizer.init_kwargs.get("_commit_hash")
    if not identity:
        identity = hashlib.sha256(
            tokenizer.backend_tokenizer.to_str().encode()
        ).hexdigest()
    expected = metadata.get("state_identity", {}).get("tokenizer")
    if expected is not None and expected != identity:
        raise ValueError("Analysis tokenizer differs from the recorded tokenizer")
    return tokenizer
