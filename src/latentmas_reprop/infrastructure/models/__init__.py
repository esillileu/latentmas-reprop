"""Model adapters. Heavy imports stay lazy so dtype checks do not load generation."""


def __getattr__(name: str):
    if name == "ModelWrapper":
        from .model_wrapper import ModelWrapper

        return ModelWrapper
    if name == "_ensure_pad_token":
        from .loader import ensure_pad_token

        return ensure_pad_token
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
