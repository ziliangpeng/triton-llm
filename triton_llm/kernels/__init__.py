"""Shared Triton kernel implementations for LLM inference.

Exports all kernels for convenient access via ``from triton_llm.kernels import ...``.
"""

from .add import add
from .attention import attention
from .attention_gqa import attention_gqa
from .embedding import embedding
from .gelu import gelu
from .gemm import gemm
from .layernorm import layer_norm
from .rms_norm import rms_norm
from .rope import precompute_cos_sin, apply_rope
from .softmax import softmax
from .swiglu import swiglu
from .transpose_2d import to_head_major_dev, to_seq_major_dev, flat_to_cache_dev, cache_to_flat_dev

__all__ = [
    "add",
    "attention",
    "attention_gqa",
    "embedding",
    "gelu",
    "gemm",
    "layer_norm",
    "rms_norm",
    "precompute_cos_sin",
    "apply_rope",
    "softmax",
    "swiglu",
    "to_head_major_dev",
    "to_seq_major_dev",
    "flat_to_cache_dev",
    "cache_to_flat_dev",
]
