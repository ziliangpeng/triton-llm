"""SmolLM2 Triton inference engine.

Pure Python + Triton (no PyTorch) implementation of SmolLM2,
a fully open Llama-architecture model by HuggingFace.
"""

from .config import SmolLM2Config
from .model import SmolLM2ForCausalLM

__all__ = ["SmolLM2Config", "SmolLM2ForCausalLM"]
