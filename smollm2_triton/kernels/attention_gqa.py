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
from gpt2_triton import gpu


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

    # --- Single tiled pass over K, V ---
    for start in range(0, seq_k, BLOCK_SIZE):
        offs_n = start + tl.arange(0, BLOCK_SIZE)
        mask_n = offs_n < seq_k

        if CAUSAL:
            k_mask_1d = mask_n & (offs_n <= abs_q_pos)
        else:
            k_mask_1d = mask_n

        k_mask = tl.broadcast_to(k_mask_1d[:, None], (BLOCK_SIZE, HEAD_SIZE))
        k_ptrs = K + k_head_offset + offs_n[:, None] * stride_k + offs_d[None, :]
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
        v_ptrs = V + v_head_offset + offs_n[:, None] * stride_v + offs_d[None, :]
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)

        # Fused update: acc = acc * rescale + p @ V_block
        acc = acc * rescale + tl.sum(p[:, None] * v, axis=0)

        # Update running statistics
        row_sum = row_sum * rescale + block_sum
        row_max = new_max

    # Normalize by the final sum
    acc = acc / row_sum

    # Store output row
    tl.store(O + pid * stride_o + offs_d, acc)


def attention_gqa(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_head: int,
    n_kv_head: int,
    causal: bool = True,
    sm_scale: float | None = None,
) -> np.ndarray:
    """Fused GQA (Grouped Query Attention): O = softmax(Q @ K^T * sm_scale) @ V.

    Input format (flat)::

        Q: (n_head * seq_q, d_k)  — rows grouped by head
        K: (n_kv_head * seq_k, d_k)
        V: (n_kv_head * seq_k, d_k)

    When ``causal=True`` (default), a causal (upper-triangular) mask is applied
    so that position *i* can only attend to positions ``j <= i``.

    Parameters
    ----------
    q : np.ndarray
        Query, shape ``(n_head * seq_q, d_k)``, float32.
    k : np.ndarray
        Key, shape ``(n_kv_head * seq_k, d_k)``, float32.
    v : np.ndarray
        Value, shape ``(n_kv_head * seq_k, d_k)``, float32.
    n_head : int
        Number of query heads.
    n_kv_head : int
        Number of key/value heads. Must divide ``n_head``.
    causal : bool
        If True, apply causal masking (default). If False, no masking.
    sm_scale : float or None
        Softmax scale. If None, defaults to ``1.0 / sqrt(d_k)``.

    Returns
    -------
    o : np.ndarray
        Output in flat format ``(n_head * seq_q, d_k)``, float32.
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
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(
            f"All inputs must be 2D arrays, got q.ndim={q.ndim}, k.ndim={k.ndim}, v.ndim={v.ndim}"
        )
    if q.shape[0] % n_head != 0:
        raise ValueError(
            f"Query shape[0] ({q.shape[0]}) must be divisible by n_head ({n_head})"
        )
    if k.shape[0] % n_kv_head != 0:
        raise ValueError(
            f"Key shape[0] ({k.shape[0]}) must be divisible by n_kv_head ({n_kv_head})"
        )

    # Input format: all inputs in FLAT format (n_head * seq, d_k).
    # The model code is responsible for reshaping before calling this function.

    # Ensure C-contiguous float32
    q = np.require(q, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    k = np.require(k, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    v = np.require(v, dtype=np.float32, requirements=["C_CONTIGUOUS"])

    d_k = q.shape[1]
    seq_q = q.shape[0] // n_head
    seq_k = k.shape[0] // n_kv_head

    # --- Validate dimensions ---
    if k.shape != (n_kv_head * seq_k, d_k):
        raise ValueError(
            f"K shape {k.shape} != expected ({n_kv_head * seq_k}, {d_k})"
        )
    if v.shape != (n_kv_head * seq_k, d_k):
        raise ValueError(
            f"V shape {v.shape} != expected ({n_kv_head * seq_k}, {d_k})"
        )
    # --- Handle empty inputs ---
    if seq_q == 0:
        return np.empty((0, d_k), dtype=np.float32)
    if seq_k == 0:
        # When there are no keys/values to attend to, output should be zeros
        return np.zeros((n_head * seq_q, d_k), dtype=np.float32)

    if d_k <= 0 or (d_k & (d_k - 1)) != 0:
        raise ValueError(
            f"d_k ({d_k}) must be a positive power of 2 for Triton compilation"
        )

    # Default sm_scale
    if sm_scale is None:
        sm_scale = 1.0 / (d_k ** 0.5)

    # --- Strides in elements (not bytes) ---
    stride_q = q.strides[0] // q.itemsize  # == d_k for contiguous
    stride_k = k.strides[0] // k.itemsize
    stride_v = v.strides[0] // v.itemsize

    # --- Move to device ---
    q_dev = gpu.to_device(q)
    k_dev = gpu.to_device(k)
    v_dev = gpu.to_device(v)
    o_dev = gpu.allocate((n_head * seq_q, d_k), np.float32)
    stride_o = o_dev.shape[1]  # row stride in elements

    BLOCK_SIZE = 64
    grid = (seq_q, n_head)  # 2D grid: (seq_pos, q_head)

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

    gpu.synchronize()
    return gpu.to_host(o_dev)
