"""
Triton GEMM Kernel (CUDA + HIP)

A cross-vendor matrix multiplication kernel using Triton.
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) via the unified gpu allocator.
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _gemm_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Cast raw pointers to proper pointer type (fixes int64 vs pointer mismatch on Triton 3.x)
    A = tl.cast(A, tl.pointer_type(tl.float32))
    B = tl.cast(B, tl.pointer_type(tl.float32))
    C = tl.cast(C, tl.pointer_type(tl.float32))

    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm(
    h_dev: "gpu.DeviceTensor",
    w_dev: "gpu.DeviceTensor",
    out_dev: "gpu.DeviceTensor | None" = None,
) -> "gpu.DeviceTensor":
    """GPU-resident GEMM. No sync, no host copies.

    Parameters
    ----------
    h_dev : DeviceTensor, shape (M, K), float32
        Hidden states on GPU.
    w_dev : DeviceTensor, shape (K, N), float32
        Weight matrix on GPU.
    out_dev : DeviceTensor, shape (M, N), float32, optional
        Pre-allocated output. Auto-allocated if None.

    Returns
    -------
    out_dev : DeviceTensor, shape (M, N), float32
        Product on GPU.
    """
    M, K = h_dev.shape
    K2, N = w_dev.shape
    assert K == K2, f"gemm_device shape mismatch: ({M},{K}) x ({K2},{N})"
    if out_dev is not None:
        assert out_dev.shape == (M, N), f"out_dev shape mismatch: expected ({M},{N}), got {out_dev.shape}"

    # Handle zero-dimension edge cases
    if K == 0 or M == 0 or N == 0:
        if out_dev is None:
            out_dev = gpu.allocate((M, N), np.float32)
        return out_dev

    if out_dev is None:
        out_dev = gpu.allocate((M, N), np.float32)

    # Contiguous strides in elements
    stride_hm, stride_hk = K, 1
    stride_wk, stride_wn = N, 1
    stride_cm, stride_cn = N, 1

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        h_dev.data_ptr(), w_dev.data_ptr(), out_dev.data_ptr(),
        M, N, K,
        stride_hm, stride_hk,
        stride_wk, stride_wn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    return out_dev