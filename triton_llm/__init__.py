"""triton_llm — Shared Triton kernel library and GPU allocator.

All Triton kernels live here, shared across model packages
(smollm2_triton, etc.).
"""

from . import gpu

__all__ = ["gpu"]
