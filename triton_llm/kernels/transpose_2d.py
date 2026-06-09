"""Triton 2D transpose and KV cache copy kernels (CUDA + HIP).

Transposes between (seq, n_heads*d_k) and (n_heads*seq, d_k) layouts,
and copies between head-major flat format and KV cache (n_kv_head, max_seq, d_k) format.
All operations are GPU-only with no host round-trips.
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _to_head_major_kernel(
    I, O,
    seq, n_heads,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy (seq, n_heads*d_k) -> (n_heads*seq, d_k).

    pid = head * seq + seq_pos
    Input:  in[seq_pos * n_heads * HEAD_SIZE + head * HEAD_SIZE + dim]
    Output: out[(head * seq + seq_pos) * HEAD_SIZE + dim]
    """
    I = tl.cast(I, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    head = pid // seq
    seq_pos = pid % seq

    in_off = seq_pos * n_heads * HEAD_SIZE + head * HEAD_SIZE
    out_off = (head * seq + seq_pos) * HEAD_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(I + in_off + offs, mask=mask, other=0.0)
    tl.store(O + out_off + offs, x, mask=mask)


@triton.jit
def _to_seq_major_kernel(
    I, O,
    seq, n_heads,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy (n_heads*seq, d_k) -> (seq, n_heads*d_k).

    pid = head * seq + seq_pos
    Input:  in[(head * seq + seq_pos) * HEAD_SIZE + dim]
    Output: out[seq_pos * n_heads * HEAD_SIZE + head * HEAD_SIZE + dim]
    """
    I = tl.cast(I, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    head = pid // seq
    seq_pos = pid % seq

    in_off = (head * seq + seq_pos) * HEAD_SIZE
    out_off = seq_pos * n_heads * HEAD_SIZE + head * HEAD_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(I + in_off + offs, mask=mask, other=0.0)
    tl.store(O + out_off + offs, x, mask=mask)


@triton.jit
def _flat_to_cache_kernel(
    FLAT, CACHE,
    n_kv_head, seq, max_seq, pos_offset,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy head-major flat (n_kv_head*seq, d_k) -> cache (n_kv_head, max_seq, d_k) at pos_offset.

    pid = h * seq + p
    Flat:  flat_off = (h * seq + p) * HEAD_SIZE
    Cache: cache_off = h * max_seq * HEAD_SIZE + (pos_offset + p) * HEAD_SIZE
    """
    FLAT = tl.cast(FLAT, tl.pointer_type(tl.float32))
    CACHE = tl.cast(CACHE, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    h = pid // seq
    p = pid % seq

    flat_off = (h * seq + p) * HEAD_SIZE
    cache_off = h * max_seq * HEAD_SIZE + (pos_offset + p) * HEAD_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(FLAT + flat_off + offs, mask=mask, other=0.0)
    tl.store(CACHE + cache_off + offs, x, mask=mask)


@triton.jit
def _cache_to_flat_kernel(
    CACHE, OUT,
    n_kv_head, total_seq, max_seq,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy cache slice (n_kv_head, max_seq, d_k) -> head-major flat (n_kv_head*total_seq, d_k).

    pid = h * total_seq + p
    Cache: cache_off = h * max_seq * HEAD_SIZE + p * HEAD_SIZE
    Flat:  out_off = (h * total_seq + p) * HEAD_SIZE
    """
    CACHE = tl.cast(CACHE, tl.pointer_type(tl.float32))
    OUT = tl.cast(OUT, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    h = pid // total_seq
    p = pid % total_seq

    cache_off = h * max_seq * HEAD_SIZE + p * HEAD_SIZE
    out_off = (h * total_seq + p) * HEAD_SIZE

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(CACHE + cache_off + offs, mask=mask, other=0.0)
    tl.store(OUT + out_off + offs, x, mask=mask)


# ---------------------------------------------------------------------------
# Public API: GPU-resident transpose and cache copy functions
# ---------------------------------------------------------------------------


def to_head_major_dev(
    x_dev: "gpu.DeviceTensor",
    n_head: int,
    seq: int,
    d_k: int,
) -> "gpu.DeviceTensor":
    """Transpose (seq, n_head*d_k) -> (n_head*seq, d_k) on GPU. No sync.

    Parameters
    ----------
    x_dev : DeviceTensor, shape (seq, n_head * d_k), float32
    n_head : int
        Number of (query) heads.
    seq : int
        Sequence length.
    d_k : int
        Head dimension.

    Returns
    -------
    out_dev : DeviceTensor, shape (n_head * seq, d_k), float32
    """
    out = gpu.allocate((n_head * seq, d_k), np.float32)
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_head * seq,)
    _to_head_major_kernel[grid](
        x_dev.data_ptr(), out.data_ptr(),
        seq, n_head, d_k, BLOCK_SIZE,
    )
    return out


def to_seq_major_dev(
    x_dev: "gpu.DeviceTensor",
    n_head: int,
    seq: int,
    d_k: int,
) -> "gpu.DeviceTensor":
    """Transpose (n_head*seq, d_k) -> (seq, n_head*d_k) on GPU. No sync.

    Parameters
    ----------
    x_dev : DeviceTensor, shape (n_head * seq, d_k), float32
    n_head : int
        Number of heads.
    seq : int
        Sequence length.
    d_k : int
        Head dimension.

    Returns
    -------
    out_dev : DeviceTensor, shape (seq, n_head * d_k), float32
    """
    out = gpu.allocate((seq, n_head * d_k), np.float32)
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_head * seq,)
    _to_seq_major_kernel[grid](
        x_dev.data_ptr(), out.data_ptr(),
        seq, n_head, d_k, BLOCK_SIZE,
    )
    return out


def flat_to_cache_dev(
    flat_dev: "gpu.DeviceTensor",
    cache_dev: "gpu.DeviceTensor",
    n_kv_head: int,
    seq: int,
    d_k: int,
    max_seq: int,
    pos_offset: int,
) -> None:
    """Copy head-major flat (n_kv_head*seq, d_k) -> cache (n_kv_head, max_seq, d_k) at pos_offset.

    No sync. No return value (cache_dev is modified in-place).

    Parameters
    ----------
    flat_dev : DeviceTensor, shape (n_kv_head * seq, d_k), float32
    cache_dev : DeviceTensor, shape (n_kv_head, max_seq, d_k), float32
    n_kv_head : int
        Number of KV heads.
    seq : int
        Number of new tokens to copy.
    d_k : int
        Head dimension.
    max_seq : int
        Maximum cache length (stride in cache's seq dimension).
    pos_offset : int
        Starting position in cache to write into.
    """
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_kv_head * seq,)
    _flat_to_cache_kernel[grid](
        flat_dev.data_ptr(), cache_dev.data_ptr(),
        n_kv_head, seq, max_seq, pos_offset, d_k, BLOCK_SIZE,
    )


def cache_to_flat_dev(
    cache_dev: "gpu.DeviceTensor",
    n_kv_head: int,
    total_seq: int,
    d_k: int,
    max_seq: int,
) -> "gpu.DeviceTensor":
    """Copy cache slice (n_kv_head, max_seq, d_k) -> head-major flat (n_kv_head*total_seq, d_k).

    No sync.

    Parameters
    ----------
    cache_dev : DeviceTensor, shape (n_kv_head, max_seq, d_k), float32
    n_kv_head : int
        Number of KV heads.
    total_seq : int
        Total sequence length to read from cache (number of positions, starting from 0).
    d_k : int
        Head dimension.
    max_seq : int
        Maximum cache length (stride in cache's seq dimension).

    Returns
    -------
    out_dev : DeviceTensor, shape (n_kv_head * total_seq, d_k), float32
    """
    out = gpu.allocate((n_kv_head * total_seq, d_k), np.float32)
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_kv_head * total_seq,)
    _cache_to_flat_kernel[grid](
        cache_dev.data_ptr(), out.data_ptr(),
        n_kv_head, total_seq, max_seq, d_k, BLOCK_SIZE,
    )
    return out
