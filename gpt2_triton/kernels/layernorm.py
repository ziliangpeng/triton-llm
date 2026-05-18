"""
Triton LayerNorm Kernel (CUDA + HIP)

A cross-vendor LayerNorm kernel using Triton.
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) via the unified gpu allocator.

LayerNorm is computed over the last dimension:
    y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

This is the standard formulation used in GPT-2 (and most Transformer models).
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


@triton.jit
def _layer_norm_kernel(
    X, Y, W, B,
    stride_x_row, stride_y_row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program instance per row.

    X: (M, N) input, Y: (M, N) output, W: (N,) gamma, B: (N,) beta.
    Mean and variance are computed across the row (last dim).
    """
    # Cast raw pointers to typed float32 pointers (Triton 3.x).
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    W = tl.cast(W, tl.pointer_type(tl.float32))
    B = tl.cast(B, tl.pointer_type(tl.float32))

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x_row_ptr = X + row * stride_x_row
    y_row_ptr = Y + row * stride_y_row

    # Load the full row (masked) into registers.
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)

    # Mean and variance over the row.
    # Use a numerically stable two-pass form: compute mean first,
    # then variance via E[(x - mean)^2]. This avoids the catastrophic
    # cancellation that E[x^2] - E[x]^2 suffers from when the row has
    # a large mean (e.g. residual-stream activations in GPT-2).
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / N

    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Affine transform with gamma / beta.
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = x_centered * rstd * w + b

    tl.store(y_row_ptr + cols, y, mask=mask)


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Apply LayerNorm to ``x`` along the last dimension.

    Parameters
    ----------
    x : (M, N) float32 array.
    gamma : (N,) float32 scale parameter.
    beta : (N,) float32 bias parameter.
    eps : float, small constant for numerical stability.

    Returns
    -------
    y : (M, N) float32 array.
    """
    assert x.ndim == 2, "LayerNorm expects a 2D (M, N) input"
    assert gamma.ndim == 1 and beta.ndim == 1, "gamma/beta must be 1D"
    M, N = x.shape
    assert gamma.shape == (N,), f"gamma must have shape ({N},), got {gamma.shape}"
    assert beta.shape == (N,), f"beta must have shape ({N},), got {beta.shape}"

    # Ensure C-contiguous float32 so device strides match what the kernel expects.
    x = np.ascontiguousarray(x, dtype=np.float32)
    gamma = np.ascontiguousarray(gamma, dtype=np.float32)
    beta = np.ascontiguousarray(beta, dtype=np.float32)

    # Strides in elements (not bytes).
    stride_x_row = x.strides[0] // x.itemsize  # == N for contiguous input
    stride_y_row = N

    x_dev = gpu.to_device(x)
    y_dev = gpu.allocate((M, N), np.float32)
    w_dev = gpu.to_device(gamma)
    b_dev = gpu.to_device(beta)

    # One row per program; block size must cover the full row.
    # Triton requires BLOCK_SIZE to be a power of two for reductions.
    BLOCK_SIZE = max(_next_pow2(N), 16)

    grid = (M,)

    _layer_norm_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(),
        w_dev.data_ptr(), b_dev.data_ptr(),
        stride_x_row, stride_y_row,
        N,
        eps,
        BLOCK_SIZE,
    )

    gpu.synchronize()
    return gpu.to_host(y_dev)
