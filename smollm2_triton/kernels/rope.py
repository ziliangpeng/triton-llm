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
from gpt2_triton import gpu


@triton.jit
def _rope_kernel(
    X,                  # raw int64 device pointer, shape (n_rows, d_k)
    cos, sin,           # raw int64 device pointers, shape (max_seq, d_k // 2) — half-packed
    n_rows,             # total number of rows
    seq_len,            # actual sequence length (determines position per row)
    stride_x,           # row stride in elements (d_k)
    position_offset,    # start position (0 for prefill, cached_len for decode)
    D_K: tl.constexpr,  # head dimension
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
    offs_half = tl.arange(0, D_K // 2)
    offs_even = offs_half            # 0, 1, ..., half-1
    offs_odd = offs_half + half      # half, half+1, ..., D_K-1

    # Load even/odd halves of the X row.
    x_even = tl.load(X + pid * stride_x + offs_even, mask=offs_even < D_K, other=0.0)
    x_odd = tl.load(X + pid * stride_x + offs_odd, mask=offs_odd < D_K, other=0.0)

    # Load cos/sin for this position (half-packed: (max_seq, d_k//2)).
    c = tl.load(cos + pos * half + offs_half, mask=offs_half < half, other=0.0)
    s = tl.load(sin + pos * half + offs_half, mask=offs_half < half, other=0.0)

    # Apply RoPE rotation to the pair halves.
    rotated_even = x_even * c - x_odd * s
    rotated_odd = x_even * s + x_odd * c

    # Store back in-place.
    tl.store(X + pid * stride_x + offs_even, rotated_even, mask=offs_even < D_K)
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


def apply_rope(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    seq_len: int,
    position_offset: int = 0,
) -> np.ndarray:
    """
    Apply Rotary Position Embedding to a single tensor (Q or K) in-place on GPU.

    Parameters
    ----------
    x : (n_rows, d_k) float32 array.
        A single Q or K tensor, already projected. Leading dimensions are
        flattened internally, so 3D (batch, n_head, seq, d_k) and higher
        shapes are all supported.
    cos : (max_seq, d_k // 2) float32 array — half-packed cos table.
    sin : (max_seq, d_k // 2) float32 array — half-packed sin table.
    seq_len : int
        The actual sequence length. Used to determine position:
        ``pos = (row % seq_len) + position_offset``.
    position_offset : int
        Position offset for KV cache decode (default: 0).

    Returns
    -------
    y : same shape and dtype as ``x``.
    """
    if x.ndim < 2:
        raise ValueError(f"apply_rope expects at least a 2D input, got {x.ndim}D")
    if x.shape[-1] < 2 or x.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension (d_k) must be even and >= 2, got {x.shape[-1]}")
    if cos.ndim != 2 or sin.ndim != 2:
        raise ValueError(f"cos and sin must be 2D arrays, got cos.ndim={cos.ndim}, sin.ndim={sin.ndim}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos and sin must have same shape, got cos.shape={cos.shape}, sin.shape={sin.shape}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if position_offset < 0:
        raise ValueError(f"position_offset must be >= 0, got {position_offset}")
    if position_offset + seq_len > cos.shape[0]:
        raise ValueError(
            f"position_offset ({position_offset}) + seq_len ({seq_len}) exceeds "
            f"cos table max_seq ({cos.shape[0]})"
        )

    orig_shape = x.shape
    d_k = orig_shape[-1]
    half = d_k // 2

    if cos.shape[-1] != half:
        raise ValueError(
            f"cos.shape[-1]={cos.shape[-1]} must equal d_k//2={half}"
        )
    if sin.shape[-1] != half:
        raise ValueError(
            f"sin.shape[-1]={sin.shape[-1]} must equal d_k//2={half}"
        )

    # Flatten all leading dimensions so the kernel sees (n_rows, d_k).
    x = np.ascontiguousarray(x.reshape(-1, d_k), dtype=np.float32)
    cos = np.ascontiguousarray(cos, dtype=np.float32)
    sin = np.ascontiguousarray(sin, dtype=np.float32)
    n_rows = x.shape[0]

    # Handle empty input (n_rows == 0): return empty array with correct shape.
    if n_rows == 0:
        return np.empty(orig_shape, dtype=np.float32)

    # Stride in elements (not bytes).
    stride_x = x.strides[0] // x.itemsize  # == d_k for contiguous input

    x_dev = gpu.to_device(x)
    cos_dev = gpu.to_device(cos)
    sin_dev = gpu.to_device(sin)

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
    )

    # Explicit device synchronize to ensure kernel completion before host read.
    gpu.synchronize()
    return gpu.to_host(x_dev).reshape(orig_shape)
