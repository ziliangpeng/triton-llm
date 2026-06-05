"""
gpt2_triton — Pure Python + Triton GPT-2 Inference (zero PyTorch at runtime).

This package provides GPU-accelerated GPT-2 inference components
using Triton kernels and a lightweight ctypes-based GPU allocator.
"""

from .config import GPT2Config
from .model import GPT2Model

__all__ = ["GPT2Config", "GPT2Model"]
