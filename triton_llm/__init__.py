"""triton_llm — Shared Triton kernel library and GPU allocator.

All Triton kernels live here, shared across model packages
(gpt2_triton, smollm2_triton, etc.).
"""

from . import gpu
from . import kernels as kernels

__all__ = ["gpu", "kernels"]
