"""Decoder implementations available for profiling."""

from collections.abc import Callable

from latentmas_reprop.application.intervention_continuation.decoder import (
    decode_tokens_with_hiddens,
)

from .buffer_variants import (
    decode_cumulative_best,
    decode_v2_gpu_buffering,
    decode_v3_gpu_token_tensor,
)
from .cache_variants import decode_v7_static_cache, decode_v9_cuda_graph
from .compiled_variant import decode_v8_torch_compile
from .model_variants import (
    decode_v1_model_inner,
    decode_v4_inference_mode,
    decode_v5_mask_optimization,
    decode_v6_last_hidden_direct,
)

VARIANTS: dict[str, Callable] = {
    "baseline": decode_tokens_with_hiddens,
    "v1_model_inner": decode_v1_model_inner,
    "v2_gpu_buffering": decode_v2_gpu_buffering,
    "v3_gpu_token_tensor": decode_v3_gpu_token_tensor,
    "v4_inference_mode": decode_v4_inference_mode,
    "v5_mask_optimization": decode_v5_mask_optimization,
    "v6_last_hidden_direct": decode_v6_last_hidden_direct,
    "v7_static_cache": decode_v7_static_cache,
    "v8_torch_compile": decode_v8_torch_compile,
    "v9_cuda_graph": decode_v9_cuda_graph,
    "cumulative_best": decode_cumulative_best,
}
