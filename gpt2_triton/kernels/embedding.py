"""Triton Embedding + Positional Encoding Kernel for GPT-2 inference (CUDA + HIP).

Combines token embedding (gather from vocabulary weight matrix) with learned
positional encoding in a single fused kernel.  Uses a 2-D grid so that large
embedding dimensions (>BLOCK_SIZE) are handled correctly.
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


@triton.jit
def _embedding_kernel(
    token_ids,         # int32 device pointer, shape (batch * seq_len,)
    weight,            # float32 device pointer, shape (vocab_size, n_embd)
    pos_weight,        # float32 device pointer, shape (max_position, n_embd)
    output,            # float32 device pointer, shape (batch * seq_len, n_embd)
    vocab_size,        # int — reserved for future bounds checking
    n_embd,            # int — embedding dimension
    seq_len,           # int — sequence length (used to compute position index)
    position_offset,   # int — absolute position offset for KV cache decode
    stride_weight,     # int — weight row stride in elements
    stride_pos,        # int — pos_weight row stride in elements
    BLOCK_SIZE: tl.constexpr,  # tile over n_embd
):
    """Fused token embedding + positional encoding for a single token position.

    Uses a 2-D grid ``(batch * seq_len, n_block)`` where each program handles
    a contiguous chunk of the embedding dimension for one position.
    The kernel::

        1. Loads ``token_id = token_ids[b, s]``
        2. Gathers ``weight[token_id, col_start:col_start+BLOCK_SIZE]``
        3. Gathers ``pos_weight[position_offset + s, col_start:col_start+BLOCK_SIZE]``
        4. Stores ``output[b, s, col_start:] = emb_chunk + pos_chunk``

    Parameters
    ----------
    token_ids : int
        Raw int64 device pointer to flattened int32 token IDs (cast inside).
    weight : int
        Raw int64 device pointer to float32 token embedding weight.
    pos_weight : int
        Raw int64 device pointer to float32 positional encoding weight.
    output : int
        Raw int64 device pointer for the float32 output.
    vocab_size : int
        Vocabulary size (reserved for future bounds checking).
    n_embd : int
        Embedding dimension.
    seq_len : int
        Sequence length (used to derive ``s = pid_x % seq_len``).
    position_offset : int
        Absolute position offset added to derive the index into pos_weight.
        For KV cache decode, this is the current sequence position so far.
    stride_weight : int
        Row stride of ``weight`` in elements (not bytes).
    stride_pos : int
        Row stride of ``pos_weight`` in elements (not bytes).
    BLOCK_SIZE : tl.constexpr
        Number of embedding dimensions processed per program.
    """
    pid_x = tl.cast(tl.program_id(0), tl.int64)  # flat index over (batch * seq_len)
    pid_y = tl.program_id(1)  # block index over n_embd dimension

    # Cast raw int64 pointers to typed pointers (Triton 3.x compatibility)
    output = tl.cast(output, tl.pointer_type(tl.float32))
    weight = tl.cast(weight, tl.pointer_type(tl.float32))
    pos_weight = tl.cast(pos_weight, tl.pointer_type(tl.float32))
    token_ids = tl.cast(token_ids, tl.pointer_type(tl.int32))

    # Position within the sequence (column index)
    s = pid_x % seq_len
    abs_pos = s + position_offset

    # Load the token ID for this position
    token_id = tl.load(token_ids + pid_x)

    # Tile over the embedding dimension
    offs = pid_y * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_embd

    # Gather token embedding: weight[token_id, offs]
    emb_ptrs = weight + tl.cast(token_id, tl.int64) * stride_weight + offs
    emb = tl.load(emb_ptrs, mask=mask, other=0.0)

    # Gather positional encoding: pos_weight[abs_pos, offs]
    pos_ptrs = pos_weight + abs_pos * stride_pos + offs
    pos = tl.load(pos_ptrs, mask=mask, other=0.0)

    # Combine and store
    out = emb + pos
    tl.store(output + pid_x * n_embd + offs, out, mask=mask)


def embedding(
    token_ids: np.ndarray,      # (batch, seq_len), int32
    weight: np.ndarray,          # (vocab_size, n_embd), float32
    pos_weight: np.ndarray,      # (max_position, n_embd), float32
    position_offset: int = 0,    # absolute position offset for KV cache decode
) -> np.ndarray:                 # (batch, seq_len, n_embd), float32
    """Fused token embedding + positional encoding.

    For every position ``(b, s)`` in the batch::

        output[b, s, :] = weight[token_ids[b, s], :] + pos_weight[s + position_offset, :]

    Parameters
    ----------
    token_ids : np.ndarray
        2-D int32 array of token IDs, shape ``(batch, seq_len)``.
    weight : np.ndarray
        2-D float32 token embedding weight, shape ``(vocab_size, n_embd)``.
    pos_weight : np.ndarray
        2-D float32 positional encoding weight, shape ``(max_position, n_embd)``.
    position_offset : int
        Absolute position offset added to each token's index into pos_weight.
        Default 0 for full-sequence embedding. For KV cache decode steps,
        set to the current total sequence length so far.

    Returns
    -------
    output : np.ndarray
        Float32 array of shape ``(batch, seq_len, n_embd)``.
    """
    assert token_ids.ndim == 2, f"token_ids must be 2-D, got shape {token_ids.shape}"
    assert weight.ndim == 2, f"weight must be 2-D, got shape {weight.shape}"
    assert pos_weight.ndim == 2, f"pos_weight must be 2-D, got shape {pos_weight.shape}"

    batch, seq_len = token_ids.shape
    vocab_size, n_embd = weight.shape
    max_position, n_embd_pos = pos_weight.shape

    assert n_embd == n_embd_pos, \
        f"Embedding dimension mismatch: weight has {n_embd}, pos_weight has {n_embd_pos}"
    assert seq_len <= max_position, \
        f"seq_len ({seq_len}) exceeds max_position ({max_position})"

    # Empty input shortcut
    if batch == 0 or seq_len == 0:
        return np.empty((batch, seq_len, n_embd), dtype=np.float32)

    # Ensure C-contiguous with correct dtypes
    token_ids = np.require(token_ids, dtype=np.int32, requirements=["C_CONTIGUOUS"])
    weight = np.require(weight, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    pos_weight = np.require(pos_weight, dtype=np.float32, requirements=["C_CONTIGUOUS"])

    # Flatten token_ids to 1-D: (batch * seq_len,)
    token_ids_flat = token_ids.reshape(-1)

    # Row strides in elements (not bytes)
    stride_weight = weight.strides[0] // weight.itemsize
    stride_pos = pos_weight.strides[0] // pos_weight.itemsize

    # Allocate output: (batch * seq_len, n_embd) — flat 2-D
    flat_shape = (batch * seq_len, n_embd)

    token_ids_dev = gpu.to_device(token_ids_flat)
    weight_dev = gpu.to_device(weight)
    pos_weight_dev = gpu.to_device(pos_weight)
    output_dev = gpu.allocate(flat_shape, np.float32)

    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_embd))
    n_block = triton.cdiv(n_embd, BLOCK_SIZE)
    grid = (batch * seq_len, n_block)

    _embedding_kernel[grid](
        token_ids_dev.data_ptr(),
        weight_dev.data_ptr(),
        pos_weight_dev.data_ptr(),
        output_dev.data_ptr(),
        vocab_size,
        n_embd,
        seq_len,
        position_offset,
        stride_weight,
        stride_pos,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    gpu.synchronize()
    result = gpu.to_host(output_dev)
    return result.reshape(batch, seq_len, n_embd)
