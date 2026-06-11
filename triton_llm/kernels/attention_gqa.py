"""Triton Grouped Query Attention (GQA) Kernel (CUDA + HIP)

Computes GQA with optional causal masking:

    O = softmax(Q @ K^T * sm_scale) @ V

where ``n_head`` query heads share ``n_kv_head`` key/value heads (n_kv_head <= n_head,
with n_head divisible by n_kv_head).  Each query head maps to exactly one KV head:

    kv_head_idx = q_head_idx // (n_head // n_kv_head)

Reference: https://arxiv.org/abs/2305.13245
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _gqa_attention_kernel(
    Q, K, V, O,
    seq_q, seq_k,
    stride_q, stride_k, stride_v, stride_o,
    n_head, n_kv_head,
    sm_scale,
    GROUP_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """Fused GQA (optionally causal) attention for a single (seq_pos, q_head) pair.

    Launched with a 2D grid of shape ``(seq_q, n_head)`` so that each program
    retrieves ``q_pos = tl.program_id(0)`` and ``q_head_idx = tl.program_id(1)``
    directly without integer division.

    Uses online softmax for numerical stability in a single tiled pass.

    Parameters
    ----------
    Q, K, V : int
        Raw int64 device pointers (cast to ``float32`` inside).
    O : int
        Raw int64 device pointer for the output.
    seq_q, seq_k : int
        Sequence lengths of Q and K/V respectively.
    stride_q, stride_k, stride_v, stride_o : int
        Row strides in elements (not bytes).
    n_head, n_kv_head : int
        Number of query heads and key/value heads.
    sm_scale : float
        Softmax scale (e.g. 1.0 / sqrt(d_k)).
    GROUP_SIZE : tl.constexpr
        ``n_head // n_kv_head``, compile-time constant for fast division.
    HEAD_SIZE : tl.constexpr
        Head dimension (d_k) as a compile-time constant.
    BLOCK_SIZE : tl.constexpr
        Number of key/value positions processed per loop iteration.
    CAUSAL : tl.constexpr
        If True, apply causal (upper-triangular) masking.
    """
    # 2D grid: q_pos = program_id(0), q_head_idx = program_id(1)
    q_pos = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    pid = q_head_idx * seq_q + q_pos  # flat row index

    # Cast raw int64 pointers to typed float32 pointers (Triton 3.x compat)
    Q = tl.cast(Q, tl.pointer_type(tl.float32))
    K = tl.cast(K, tl.pointer_type(tl.float32))
    V = tl.cast(V, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float32))

    # GQA routing: which KV head this Q head maps to
    kv_head_idx = q_head_idx // GROUP_SIZE

    # Load the query row
    offs_d = tl.arange(0, HEAD_SIZE)
    q = tl.load(Q + pid * stride_q + offs_d)

    # Scale q by sm_scale
    q = q * sm_scale

    # --- Online softmax accumulators ---
    acc = tl.zeros((HEAD_SIZE,), dtype=tl.float32)
    row_max = -float("inf")
    row_sum = 0.0

    # KV head base offset in elements
    k_head_offset = kv_head_idx * seq_k * stride_k
    v_head_offset = kv_head_idx * seq_k * stride_v

    # Absolute position of this query within the full sequence.
    # Prefill:  seq_k == seq_q, so abs_q_pos = q_pos.
    # Decode:   seq_q == 1, seq_k > seq_q, so abs_q_pos = seq_k - 1.
    abs_q_pos = seq_k - seq_q + q_pos

    # Pre-compute tile-level pointer base to hoist per-iteration 2D arithmetic
    offs_n_init = tl.arange(0, BLOCK_SIZE)
    k_ptrs = K + k_head_offset + offs_n_init[:, None] * stride_k + offs_d[None, :]
    v_ptrs = V + v_head_offset + offs_n_init[:, None] * stride_v + offs_d[None, :]

    # --- Single tiled pass over K, V ---
    for start in range(0, seq_k, BLOCK_SIZE):
        offs_n = start + offs_n_init
        mask_n = offs_n < seq_k

        if CAUSAL:
            k_mask_1d = mask_n & (offs_n <= abs_q_pos)
        else:
            k_mask_1d = mask_n

        k_mask = tl.broadcast_to(k_mask_1d[:, None], (BLOCK_SIZE, HEAD_SIZE))
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Attention scores: s = q_scaled @ k^T
        s = tl.sum(q[None, :] * k, axis=1)
        s = tl.where(k_mask_1d, s, -float("inf"))

        # Online softmax: update running max and rescale
        block_max = tl.max(s, axis=0)
        new_max = tl.maximum(row_max, block_max)
        rescale = tl.exp(row_max - new_max)
        p = tl.exp(s - new_max)
        block_sum = tl.sum(p, axis=0)

        # Load V block: (BLOCK_SIZE, HEAD_SIZE)
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Fused update: acc = acc * rescale + p @ V_block
        acc = acc * rescale + tl.sum(p[:, None] * v, axis=0)

        # Update running statistics
        row_sum = row_sum * rescale + block_sum
        row_max = new_max

        # Advance pointers to the next tile
        k_ptrs += BLOCK_SIZE * stride_k
        v_ptrs += BLOCK_SIZE * stride_v

    # Normalize by the final sum
    acc = acc / row_sum

    # Store output row
    tl.store(O + pid * stride_o + offs_d, acc)


def attention_gqa(
    q_dev: "gpu.DeviceTensor",
    k_dev: "gpu.DeviceTensor",
    v_dev: "gpu.DeviceTensor",
    n_head: int,
    n_kv_head: int,
    causal: bool = True,
    sm_scale: float | None = None,
) -> "gpu.DeviceTensor":
    """GPU-resident GQA attention. No sync, no host copies.

    Input format (flat)::

        Q: (n_head * seq_q, d_k)  — rows grouped by head
        K: (n_kv_head * seq_k, d_k)
        V: (n_kv_head * seq_k, d_k)

    Parameters
    ----------
    q_dev : DeviceTensor, shape (n_head * seq_q, d_k), float32
        Query on GPU.
    k_dev : DeviceTensor, shape (n_kv_head * seq_k, d_k), float32
        Key on GPU.
    v_dev : DeviceTensor, shape (n_kv_head * seq_k, d_k), float32
        Value on GPU.
    n_head : int
        Number of query heads.
    n_kv_head : int
        Number of key/value heads. Must divide ``n_head``.
    causal : bool
        If True, apply causal masking (default).
    sm_scale : float or None
        Softmax scale. If None, defaults to ``1.0 / sqrt(d_k)``.

    Returns
    -------
    o_dev : DeviceTensor, shape (n_head * seq_q, d_k), float32
        Attention output on GPU.
    """
    # --- Input validation ---
    if n_head <= 0:
        raise ValueError(f"n_head must be > 0, got {n_head}")
    if n_kv_head <= 0:
        raise ValueError(f"n_kv_head must be > 0, got {n_kv_head}")
    if n_head % n_kv_head != 0:
        raise ValueError(
            f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
        )
    if len(q_dev.shape) != 2 or len(k_dev.shape) != 2 or len(v_dev.shape) != 2:
        raise ValueError(
            f"All inputs must be 2D, got q.ndim={len(q_dev.shape)}, k.ndim={len(k_dev.shape)}, v.ndim={len(v_dev.shape)}"
        )

    # Use .shape which is a tuple for DeviceTensor
    if q_dev.shape[0] % n_head != 0:
        raise ValueError(
            f"Query shape[0] ({q_dev.shape[0]}) must be divisible by n_head ({n_head})"
        )
    if k_dev.shape[0] % n_kv_head != 0:
        raise ValueError(
            f"Key shape[0] ({k_dev.shape[0]}) must be divisible by n_kv_head ({n_kv_head})"
        )

    d_k = q_dev.shape[1]
    seq_q = q_dev.shape[0] // n_head
    seq_k = k_dev.shape[0] // n_kv_head

    if k_dev.shape != (n_kv_head * seq_k, d_k):
        raise ValueError(
            f"K shape {k_dev.shape} != expected ({n_kv_head * seq_k}, {d_k})"
        )
    if v_dev.shape != (n_kv_head * seq_k, d_k):
        raise ValueError(
            f"V shape {v_dev.shape} != expected ({n_kv_head * seq_k}, {d_k})"
        )

    if seq_q == 0:
        return gpu.allocate((0, d_k), np.float32)
    if seq_k == 0:
        return gpu.allocate((n_head * seq_q, d_k), np.float32)

    if d_k <= 0 or (d_k & (d_k - 1)) != 0:
        raise ValueError(
            f"d_k ({d_k}) must be a positive power of 2 for Triton compilation"
        )

    if sm_scale is None:
        sm_scale = 1.0 / (d_k ** 0.5)

    # Strides in elements
    stride_q = d_k  # contiguous
    stride_k = d_k
    stride_v = d_k

    o_dev = gpu.allocate((n_head * seq_q, d_k), np.float32)
    stride_o = o_dev.shape[1]

    BLOCK_SIZE = 64
    grid = (seq_q, n_head)

    _gqa_attention_kernel[grid](
        q_dev.data_ptr(),
        k_dev.data_ptr(),
        v_dev.data_ptr(),
        o_dev.data_ptr(),
        seq_q,
        seq_k,
        stride_q,
        stride_k,
        stride_v,
        stride_o,
        n_head,
        n_kv_head,
        sm_scale,
        GROUP_SIZE=n_head // n_kv_head,
        HEAD_SIZE=d_k,
        BLOCK_SIZE=BLOCK_SIZE,
        CAUSAL=causal,
    )

    return o_dev
