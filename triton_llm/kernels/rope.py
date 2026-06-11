"""
Triton RoPE (Rotary Position Embedding) Kernel (CUDA + HIP)

A cross-vendor RoPE kernel using Triton.
Works on both NVIDIA (CUDA) and AMD (HIP/ROCm) via the unified gpu allocator.

RoPE applies a rotation to query and key vectors based on their absolute
position in the sequence, using precomputed cos/sin frequency tables.

For each position p and head dimension pair (2i, 2i+1):
    theta_i = rope_theta ^ (-2i / d_k)
    cos_pi = cos(p * theta_i), sin_pi = sin(p * theta_i)
    x_rotated[2i]   = x[2i] * cos_pi - x[2i+1] * sin_pi
    x_rotated[2i+1] = x[2i] * sin_pi + x[2i+1] * cos_pi

Reference: https://arxiv.org/abs/2104.09864
"""

import numpy as np
import triton
import triton.language as tl
from triton_llm import gpu


@triton.jit
def _rope_kernel(
    X,                  # raw int64 device pointer, shape (n_rows, d_k)
    cos, sin,           # raw int64 device pointers, shape (max_seq, d_k // 2) — half-packed
    n_rows,             # total number of rows
    seq_len,            # actual sequence length (determines position per row)
    stride_x,           # row stride in elements (d_k)
    position_offset,    # start position (0 for prefill, cached_len for decode)
    D_K: tl.constexpr,  # head dimension
    BLOCK_SIZE: tl.constexpr,  # next power of 2 of D_K // 2
):
    """
    One program instance per row. Applies RoPE in-place on X.

    Each row corresponds to one (batch, head, seq) position in the flattened
    Q or K tensor. Position is determined by ``(pid % seq_len) + position_offset``,
    so batch and head indices don't affect rotation -- only position matters.
    """
    # Cast raw pointers to typed float32 pointers (Triton 3.x compat).
    X = tl.cast(X, tl.pointer_type(tl.float32))
    cos = tl.cast(cos, tl.pointer_type(tl.float32))
    sin = tl.cast(sin, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    pos = (pid % seq_len) + position_offset  # absolute position in sequence

    half = D_K // 2
    offs_half = tl.arange(0, BLOCK_SIZE)
    offs_even = offs_half            # 0, 1, ..., BLOCK_SIZE-1
    offs_odd = offs_half + half      # half, half+1, ..., half+BLOCK_SIZE-1

    # Load even/odd halves of the X row.
    x_even = tl.load(X + pid * stride_x + offs_even, mask=offs_even < half, other=0.0)
    x_odd = tl.load(X + pid * stride_x + offs_odd, mask=offs_odd < D_K, other=0.0)

    # Load cos/sin for this position (half-packed: (max_seq, d_k//2)).
    c = tl.load(cos + pos * half + offs_half, mask=offs_half < half, other=0.0)
    s = tl.load(sin + pos * half + offs_half, mask=offs_half < half, other=0.0)

    # Apply RoPE rotation to the pair halves.
    rotated_even = x_even * c - x_odd * s
    rotated_odd = x_even * s + x_odd * c

    # Store back in-place.
    tl.store(X + pid * stride_x + offs_even, rotated_even, mask=offs_even < half)
    tl.store(X + pid * stride_x + offs_odd, rotated_odd, mask=offs_odd < D_K)


def precompute_cos_sin(
    max_seq_len: int,
    d_k: int,
    theta: float = 100000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute cos and sin tables for RoPE.

    Returns (cos, sin), each of shape ``(max_seq_len, d_k // 2)`` -- half-packed,
    storing one value per pair index i.

    For position p and pair index i (0 <= i < d_k // 2):
        theta_i = 1.0 / (theta ** (2 * i / d_k))
        cos[p, i] = cos(p * theta_i)
        sin[p, i] = sin(p * theta_i)

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length (context window).
    d_k : int
        Head dimension. Must be even.
    theta : float
        Base frequency for RoPE (default: 10000.0, SmolLM2 uses 100000.0).

    Returns
    -------
    cos : np.ndarray, shape (max_seq_len, d_k // 2), float32
    sin : np.ndarray, shape (max_seq_len, d_k // 2), float32
    """
    if max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {max_seq_len}")
    if d_k < 2 or d_k % 2 != 0:
        raise ValueError(f"d_k must be even and >= 2, got {d_k}")
    if theta <= 0:
        raise ValueError(f"theta must be > 0, got {theta}")

    half = d_k // 2
    positions = np.arange(max_seq_len, dtype=np.float32)
    indices = np.arange(half, dtype=np.float32)

    # theta_i = 1.0 / (theta ** (2 * i / d_k))
    freqs = 1.0 / (theta ** (2 * indices / d_k))  # (half,)

    # angles = p * theta_i for all positions and pair indices
    angles = np.outer(positions, freqs)  # (max_seq_len, half)

    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)

    return cos, sin


def precompute_cos_sin_device(
    max_seq_len: int,
    d_k: int,
    theta: float = 100000.0,
) -> tuple["gpu.DeviceTensor", "gpu.DeviceTensor"]:
    """Precompute cos/sin tables for RoPE and keep them on GPU.

    Returns (cos_dev, sin_dev), each of shape ``(max_seq_len, d_k // 2)``, float32 on GPU.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length (context window).
    d_k : int
        Head dimension. Must be even.
    theta : float
        Base frequency for RoPE.

    Returns
    -------
    cos_dev : DeviceTensor, shape (max_seq_len, d_k // 2), float32
    sin_dev : DeviceTensor, shape (max_seq_len, d_k // 2), float32
    """
    cos_np, sin_np = precompute_cos_sin(max_seq_len, d_k, theta)
    return gpu.to_device(cos_np), gpu.to_device(sin_np)


def apply_rope(
    x_dev: "gpu.DeviceTensor",
    cos_dev: "gpu.DeviceTensor",
    sin_dev: "gpu.DeviceTensor",
    seq_len: int,
    position_offset: int = 0,
) -> "gpu.DeviceTensor":
    """GPU-resident RoPE. No sync, no host copies. Applies RoPE in-place on x_dev.

    Parameters
    ----------
    x_dev : DeviceTensor, shape (n_rows, d_k), float32
        Q or K tensor on GPU. Will be modified in-place.
    cos_dev : DeviceTensor, shape (max_seq, d_k // 2), float32
        Cos table on GPU.
    sin_dev : DeviceTensor, shape (max_seq, d_k // 2), float32
        Sin table on GPU.
    seq_len : int
        Actual sequence length.
    position_offset : int
        Position offset for KV cache decode (default: 0).

    Returns
    -------
    x_dev : DeviceTensor (same as input, modified in-place)
    """
    n_rows, d_k = x_dev.shape
    if d_k < 2 or d_k % 2 != 0:
        raise ValueError(f"d_k must be even and >= 2, got {d_k}")
    half = d_k // 2
    if len(cos_dev.shape) != 2 or len(sin_dev.shape) != 2:
        raise ValueError("cos_dev and sin_dev must be 2D tensors")
    if cos_dev.shape != sin_dev.shape:
        raise ValueError(f"cos_dev and sin_dev shape mismatch: {cos_dev.shape} vs {sin_dev.shape}")
    if cos_dev.shape[1] != half:
        raise ValueError(f"cos_dev last dim must be d_k // 2 ({half}), got {cos_dev.shape[1]}")
    if n_rows == 0:
        return x_dev
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if position_offset < 0:
        raise ValueError(f"position_offset must be >= 0, got {position_offset}")
    if position_offset + seq_len > cos_dev.shape[0]:
        raise ValueError(
            f"position_offset ({position_offset}) + seq_len ({seq_len}) "
            f"exceeds cos/sin table size ({cos_dev.shape[0]})"
        )
    if n_rows % seq_len != 0:
        raise ValueError(
            f"Total rows ({n_rows}) must be a multiple of seq_len ({seq_len})"
        )

    block_size = triton.next_power_of_2(half)
    stride_x = d_k  # contiguous row stride in elements

    grid = (n_rows,)

    _rope_kernel[grid](
        x_dev.data_ptr(),
        cos_dev.data_ptr(),
        sin_dev.data_ptr(),
        n_rows,
        seq_len,
        stride_x,
        position_offset,
        d_k,
        block_size,
    )

    return x_dev
