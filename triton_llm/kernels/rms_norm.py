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
    x: np.ndarray,
    weight: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Apply RMSNorm to ``x`` along the last dimension.

    Parameters
    ----------
    x : (..., N) float32 array. Leading dimensions are flattened internally,
        so 2D (M, N), 3D (B, S, N), and higher are all supported.
    weight : (N,) float32 scale parameter.
    eps : float, small constant for numerical stability.

    Returns
    -------
    y : same shape as ``x``.
    """
    if x.ndim < 2:
        raise ValueError(f"RMSNorm expects at least a 2D input, got {x.ndim}D")
    if weight.ndim != 1:
        raise ValueError(f"weight must be a 1D array, got {weight.ndim}D")

    orig_shape = x.shape
    N = orig_shape[-1]

    if weight.shape != (N,):
        raise ValueError(f"weight shape mismatch: expected ({N},), got {weight.shape}")
    if N == 0:
        raise ValueError("RMSNorm requires the last dimension N > 0")

    # Flatten all leading dimensions so the kernel sees (M, N).
    x = np.ascontiguousarray(x.reshape(-1, N), dtype=np.float32)
    weight = np.ascontiguousarray(weight, dtype=np.float32)
    M = x.shape[0]

    # Handle empty input (M=0): return empty array with correct shape.
    if M == 0:
        return np.empty(orig_shape, dtype=np.float32)

    # Strides in elements (not bytes).
    stride_x = x.strides[0] // x.itemsize  # == N for contiguous input
    stride_y = N

    x_dev = gpu.to_device(x)
    y_dev = gpu.allocate((M, N), np.float32)
    w_dev = gpu.to_device(weight)

    # One row per program; block size must cover the full row.
    # Triton requires BLOCK_SIZE to be a power of two for reductions.
    BLOCK_SIZE = max(_next_pow2(N), 16)

    grid = (M,)

    _rms_norm_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(), w_dev.data_ptr(),
        M, N,
        stride_x, stride_y,
        eps,
        BLOCK_SIZE,
    )

    # Explicit device synchronize to ensure kernel completion before host read.
    gpu.synchronize()
    return gpu.to_host(y_dev).reshape(orig_shape)


def rms_norm_device(
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
