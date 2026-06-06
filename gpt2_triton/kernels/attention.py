"""
Triton Fused Self-Attention Kernel (CUDA + HIP)

Computes the core GPT-2 masked self-attention operation:

    attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

with optional causal (upper-triangular) masking applied inside the kernel.
Uses online softmax for numerical stability in a single pass over K, V.
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


@triton.jit
def _attention_kernel(
    Q, K, V, O,
    N,
    stride_q, stride_k, stride_v, stride_o,
    BLOCK_SIZE: tl.constexpr,
    D_K: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Fused (optionally causal) self-attention for a single query row.

    Each program handles one row of Q.  Computes::

        O[i, :] = softmax(Q[i, :] @ K^T / sqrt(d_k)) @ V

    When ``CAUSAL=True``, positions ``j > i`` receive ``-inf`` before softmax.
    When ``CAUSAL=False``, all key/value positions are attended to (no mask).

    Numerical stability is achieved via online softmax — a single tiled pass
    that maintains a running ``max`` and ``sum(exp(x - max))``, rescaling
    the accumulator whenever a new tile has a higher max.

    Parameters
    ----------
    Q, K, V : int
        Raw int64 device pointers (cast to ``float32`` inside).
    O : int
        Raw int64 device pointer for the output.
    N : int
        Number of key/value positions (sequence length of K, V).
    stride_q, stride_k, stride_v, stride_o : int
        Row strides in elements (not bytes).
    BLOCK_SIZE : tl.constexpr
        Number of key/value positions processed per loop iteration.
    D_K : tl.constexpr
        Head dimension as a compile-time constant for performance.
    CAUSAL : tl.constexpr
        If True, apply causal (upper-triangular) masking.  If False, no mask.
    """
    row_idx = tl.program_id(0)

    # Cast raw int64 pointers to typed float32 pointers (Triton 3.x compat)
    Q = tl.cast(Q, tl.pointer_type(tl.float32))
    K = tl.cast(K, tl.pointer_type(tl.float32))
    V = tl.cast(V, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float32))

    # Load the query row: Q[row_idx, :]
    offs_d = tl.arange(0, D_K)
    q = tl.load(Q + row_idx * stride_q + offs_d)

    # Scale q at compile time -- avoids a runtime rsqrt per tile
    # (dot product linearity: s = (q * scale) @ k^T is equivalent to (q @ k^T) * scale)
    q = q * (1.0 / D_K ** 0.5)

    # --- Online softmax accumulators ---
    acc = tl.zeros((D_K,), dtype=tl.float32)   # weighted sum of V rows
    row_max = -float("inf")                     # running max
    row_sum = 0.0                               # running sum of exp(x - max)

    # --- Single tiled pass over K, V ---
    # Note: Triton does not support ``break`` in jitted functions, so we iterate
    # over all blocks and rely on masking instead.
    for start in range(0, N, BLOCK_SIZE):
        offs_n = start + tl.arange(0, BLOCK_SIZE)
        mask_n = offs_n < N

        if CAUSAL:
            # Load K block: (BLOCK_SIZE, D_K)
            # Only load elements within the causal boundary (offs_n <= row_idx)
            # to avoid wasted global memory bandwidth on masked-out positions.
            k_causal = offs_n <= row_idx
            k_mask_1d = mask_n & k_causal
            # Broadcast to (BLOCK_SIZE, D_K) -- match pointer block shape exactly.
            k_mask = tl.broadcast_to(k_mask_1d[:, None], (BLOCK_SIZE, D_K))
            k_ptrs = K + offs_n[:, None] * stride_k + offs_d[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Attention scores: s = q_scaled @ k^T
            s = tl.sum(q[None, :] * k, axis=1)

            # Causal mask: positions after row_idx get -inf
            s = tl.where(k_causal, s, -float("inf"))
        else:
            # Load ALL K/V positions -- no causal gating
            k_mask_1d = mask_n
            k_mask = tl.broadcast_to(k_mask_1d[:, None], (BLOCK_SIZE, D_K))
            k_ptrs = K + offs_n[:, None] * stride_k + offs_d[None, :]
            k = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # Attention scores: s = q_scaled @ k^T (no causal mask)
            s = tl.sum(q[None, :] * k, axis=1)

        # Online softmax: update running max and rescale
        block_max = tl.max(s, axis=0)
        new_max = tl.maximum(row_max, block_max)
        rescale = tl.exp(row_max - new_max)
        p = tl.exp(s - new_max)
        block_sum = tl.sum(p, axis=0)

        # Load V block: (BLOCK_SIZE, D_K)
        v_ptrs = V + offs_n[:, None] * stride_v + offs_d[None, :]
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Fused update: acc = acc * rescale + p @ V_block
        # p[:, None] * v: (BLOCK_SIZE, D_K), sum over axis=0: (D_K,)
        acc = acc * rescale + tl.sum(p[:, None] * v, axis=0)

        # Update running statistics
        row_sum = row_sum * rescale + block_sum
        row_max = new_max

    # Normalize by the final sum
    acc = acc / row_sum

    # Store output row
    tl.store(O + row_idx * stride_o + offs_d, acc)


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, causal: bool = True) -> np.ndarray:
    """Fused (optionally causal) self-attention: O = softmax(Q @ K^T / sqrt(d_k)) @ V.

    All inputs are 2D float32 arrays.  When ``causal=True`` (default), a causal
    (upper-triangular) mask is applied so that position *i* can only attend to
    positions ``j <= i``.  When ``causal=False``, no mask is applied, and Q may
    have a different sequence length than K/V (e.g. single-query decode with
    a larger KV cache).

    Parameters
    ----------
    q : np.ndarray
        Query, shape ``(seq_q, d_k)``, float32.
    k : np.ndarray
        Key, shape ``(seq_k, d_k)``, float32.
    v : np.ndarray
        Value, shape ``(seq_k, d_k)``, float32.
    causal : bool
        If True, apply causal masking (default).  If False, no masking.

    Returns
    -------
    o : np.ndarray
        Output, shape ``(seq_q, d_k)``, float32.
    """
    assert q.ndim == 2 and k.ndim == 2 and v.ndim == 2
    d_k = q.shape[1]
    seq_q = q.shape[0]
    seq_k = k.shape[0]
    assert k.shape == (seq_k, d_k), f"K shape {k.shape} != {({seq_k}, {d_k})}"
    assert v.shape == (seq_k, d_k), f"V shape {v.shape} != {({seq_k}, {d_k})}"
    if not causal:
        # When non-causal, Q and K/V may have different seq lengths
        assert q.shape[1] == k.shape[1] == d_k, \
            f"d_k mismatch: q={q.shape[1]}, k={k.shape[1]}, v={v.shape[1]}"
    else:
        assert seq_q == seq_k, \
            f"causal requires seq_q == seq_k, got {seq_q} vs {seq_k}"
    assert d_k > 0 and (d_k & (d_k - 1)) == 0, \
        f"d_k ({d_k}) must be a positive power of 2 for Triton compilation"

    if seq_q == 0 or seq_k == 0:
        return np.empty((seq_q, d_k), dtype=np.float32)

    # Ensure C-contiguous float32
    q = np.require(q, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    k = np.require(k, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    v = np.require(v, dtype=np.float32, requirements=["C_CONTIGUOUS"])

    # Row strides in elements (not bytes)
    stride_q = q.strides[0] // q.itemsize
    stride_k = k.strides[0] // k.itemsize
    stride_v = v.strides[0] // v.itemsize

    q_dev = gpu.to_device(q)
    k_dev = gpu.to_device(k)
    v_dev = gpu.to_device(v)
    o_dev = gpu.allocate((seq_q, d_k), np.float32)
    stride_o = o_dev.shape[1]  # row stride in elements

    # Tile size over the key/value sequence dimension
    BLOCK_SIZE = 64

    grid = (seq_q,)  # one program per query row

    _attention_kernel[grid](
        q_dev.data_ptr(), k_dev.data_ptr(), v_dev.data_ptr(), o_dev.data_ptr(),
        seq_k,
        stride_q, stride_k, stride_v, stride_o,
        BLOCK_SIZE=BLOCK_SIZE,
        D_K=d_k,
        CAUSAL=causal,
    )

    gpu.synchronize()
    return gpu.to_host(o_dev)
