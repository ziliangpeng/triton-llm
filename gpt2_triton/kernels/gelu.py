"""
Triton GELU Kernel (CUDA + HIP)

Element-wise tanh-approximation GELU activation function.
Follows the GPU allocator pattern from gemm.py / layernorm.py:
  to_device → launch → synchronize → to_host

GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


@triton.jit
def _tanh(x):
    """Numerically stable tanh via exp (Triton 3.x compatibility: no tl.tanh)."""
    x = tl.clamp(x, -20.0, 20.0)
    exp_2x = tl.exp(2.0 * x)
    return (exp_2x - 1.0) / (exp_2x + 1.0)


@triton.jit
def _gelu_kernel(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Cast raw int64 pointers to typed float32 pointers (Triton 3.x compatibility).
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask)

    # Tanh approximation of GELU
    sqrt_2_over_pi = 0.7978845608028654
    y = 0.5 * x * (1.0 + _tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))

    tl.store(Y + offsets, y, mask=mask)


def gelu(x: np.ndarray) -> np.ndarray:
    """
    Apply tanh-approximation GELU element-wise.

    Parameters
    ----------
    x : np.ndarray, float32. Any shape, processed in flattened form.

    Returns
    -------
    y : np.ndarray, same shape as ``x``.
    """
    if x.size == 0:
        return np.empty_like(x)

    x_dev = gpu.to_device(np.ascontiguousarray(x).astype(np.float32))
    y_dev = gpu.allocate(x.shape, np.float32)
    N = x.size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _gelu_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(),
        N, BLOCK_SIZE,
    )

    gpu.synchronize()
    return gpu.to_host(y_dev).reshape(x.shape)
