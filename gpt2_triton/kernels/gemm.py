"""
Triton GEMM Kernel (CUDA + HIP)

A cross-vendor matrix multiplication kernel using Triton.
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) via the unified gpu allocator.
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


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


def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Perform matrix multiplication C = A @ B using Triton.
    A: (M, K), B: (K, N) -> C: (M, N)

    Note: Input arrays are converted to C-contiguous to ensure
    correct memory layout when transferred to the device.
    Strides are derived dynamically from the actual array metadata.
    """
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"

    # Ensure C-contiguous to avoid data corruption on device
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)

    # Derive strides from actual array metadata (in elements, not bytes)
    stride_am, stride_ak = a.strides[0] // a.itemsize, a.strides[1] // a.itemsize
    stride_bk, stride_bn = b.strides[0] // b.itemsize, b.strides[1] // b.itemsize

    a_dev = gpu.to_device(a)
    b_dev = gpu.to_device(b)
    c_dev = gpu.allocate((M, N), np.float32)

    # For output C we know it's contiguous
    stride_cm, stride_cn = N, 1

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _gemm_kernel[grid](
        a_dev.data_ptr(), b_dev.data_ptr(), c_dev.data_ptr(),
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )

    gpu.synchronize()
    return gpu.to_host(c_dev)