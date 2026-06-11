"""
Triton RMSNorm Kernel (CUDA + HIP)

A cross-vendor RMSNorm kernel using Triton.
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) via the unified gpu allocator.

RMSNorm is computed over the last dimension:
    y = x / sqrt(mean(x^2) + eps) * weight

This is the standard formulation used in Llama-architecture models (SmolLM2).

Reference: https://arxiv.org/abs/1910.07467
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _rms_norm_kernel(
    X, Y, W,
    M, N,
    stride_x, stride_y,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program instance per row.

    X: (M, N) input, Y: (M, N) output, W: (N,) weight.
    RMS = sqrt(mean(x^2) + eps) is computed across the row (last dim).
    """
    # Cast raw pointers to typed float32 pointers (Triton 3.x compat).
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    W = tl.cast(W, tl.pointer_type(tl.float32))

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # Load the full row (masked) into registers.
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # Compute RMS: sqrt(mean(x^2) + eps)
    # Use explicit sum / N because tl.mean divides by BLOCK_SIZE, not N.
    # Out-of-bounds lanes are 0.0 via other=0.0, so x*x for them is 0.
    x_sq = tl.where(mask, x * x, 0.0)
    rms = tl.sqrt(tl.sum(x_sq, axis=0) / N + eps)

    # Load weight and apply.
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (x / rms) * w

    tl.store(Y + row * stride_y + cols, y, mask=mask)


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1)."""
    p = 1
    while p < n:
        p <<= 1
    return p


def rms_norm(
    x_dev: "gpu.DeviceTensor",
    w_dev: "gpu.DeviceTensor",
    out_dev: "gpu.DeviceTensor",
    eps: float = 1e-5,
) -> "gpu.DeviceTensor":
    """GPU-resident RMSNorm. No sync, no host copies.

    Parameters
    ----------
    x_dev : DeviceTensor, shape (M, N), float32
        Input on GPU.
    w_dev : DeviceTensor, shape (N,), float32
        Weight on GPU.
    out_dev : DeviceTensor, shape (M, N), float32
        Pre-allocated output on GPU.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    out_dev : DeviceTensor, shape (M, N), float32
    """
    M, N = x_dev.shape
    assert w_dev.shape == (N,), f"w_dev shape mismatch: expected ({N},), got {w_dev.shape}"
    assert out_dev.shape == (M, N), f"out_dev shape mismatch: expected ({M},{N}), got {out_dev.shape}"
    if M == 0:
        return out_dev
    if N == 0:
        raise ValueError("RMSNorm requires the last dimension N > 0")

    # Strides in elements.
    stride_x = N  # contiguous
    stride_y = N

    BLOCK_SIZE = max(_next_pow2(N), 16)

    grid = (M,)

    _rms_norm_kernel[grid](
        x_dev.data_ptr(), out_dev.data_ptr(), w_dev.data_ptr(),
        M, N,
        stride_x, stride_y,
        eps,
        BLOCK_SIZE,
    )

    return out_dev
