"""Triton Softmax Kernel (CUDA + HIP) — numerically stable, fusion-optimized.

Two specialized code paths selected at Triton JIT compile time:
  Single-block (N <= BLOCK_SIZE): 1 global load, 2-pass in registers.
  Multi-block  (N >  BLOCK_SIZE): tiled 3-pass for arbitrarily large rows.
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _softmax_kernel(
    X,
    Y,
    M,
    N,
    stride_x,
    stride_y,
    BLOCK_SIZE: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
):
    """Numerically stable softmax on a single matrix row.

    Uses two specialized code paths selected at JIT compile time:
    - ``SINGLE_BLOCK=True`` (N <= BLOCK_SIZE): single global load, 2-pass in registers.
    - ``SINGLE_BLOCK=False`` (N > BLOCK_SIZE): tiled 3-pass for arbitrarily large rows.

    Parameters
    ----------
    X : int
        Raw int64 device pointer for the input (cast to ``float32`` inside).
    Y : int
        Raw int64 device pointer for the output (cast to ``float32`` inside).
    M : int
        Number of rows (unused inside kernel, kept for interface symmetry).
    N : int
        Number of columns per row.
    stride_x : int
        Row stride of ``X`` in elements.
    stride_y : int
        Row stride of ``Y`` in elements.
    BLOCK_SIZE : tl.constexpr
        Number of columns processed per loop iteration.
    SINGLE_BLOCK : tl.constexpr
        If True, the row fits in BLOCK_SIZE — use single-load fast path.
        If False, fall back to tiled 3-pass for multi-block rows.
    """
    row_idx = tl.program_id(0)
    # Cast raw pointers to typed float32 pointers before arithmetic
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    x_ptr = X + row_idx * stride_x
    y_ptr = Y + row_idx * stride_y

    offs = tl.arange(0, BLOCK_SIZE)
    cols = offs
    mask = cols < N

    if SINGLE_BLOCK:
        # --- Fast path: single global load, 2-pass in registers ---
        # Load the row once into registers.
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))

        # Pass 1: compute max and sum-of-exp from the same loaded data.
        row_max = tl.max(x, axis=0)
        e = tl.exp(x - row_max)
        row_sum = tl.sum(e, axis=0)

        # Store result.
        y = e / row_sum
        tl.store(y_ptr + cols, y, mask=mask)
    else:
        # --- Tiled 2-pass online reduction: for rows larger than BLOCK_SIZE ---
        # Pass 1: online softmax — compute max and exp-sum simultaneously.
        #   When a new tile has a higher max, the running sum is rescaled by
        #   exp(old_max - new_max) before adding the new tile's contributions.
        #   ⚠️  Guard against NaN: when row_max == -inf the rescale factor
        #      is set to 1.0 (instead of exp(NaN)), and when new_max == -inf
        #      the tile's contribution is 0.0 (instead of exp(NaN) → NaN).
        row_max = -float("inf")
        row_sum = 0.0
        for start in range(0, N, BLOCK_SIZE):
            cols = start + offs
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(row_max, block_max)
            rescale = tl.where(row_max == -float("inf"), 1.0, tl.exp(row_max - new_max))
            block_sum = tl.where(new_max == -float("inf"), 0.0, tl.sum(tl.exp(x - new_max), axis=0))
            row_sum = row_sum * rescale + block_sum
            row_max = new_max

        # Pass 2: compute softmax values and store
        for start in range(0, N, BLOCK_SIZE):
            cols = start + offs
            mask = cols < N
            x = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
            y = tl.exp(x - row_max) / row_sum
            tl.store(y_ptr + cols, y, mask=mask)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along the last dimension (axis=-1 or 1).

    Uses a two-pass Triton kernel: first computes the row-wise maximum,
    then computes ``exp(x - max) / sum(exp(x - max))``.

    Parameters
    ----------
    x : np.ndarray
        Input array. Converted to C-contiguous float32 internally.
        - 2D input with ``axis=-1`` or ``axis=1``: softmax along rows.
        - 1D input: reshaped to ``(1, N)``, softmax applied, result flattened.
    axis : int
        Axis along which to apply softmax. Only ``-1`` and ``1`` are supported
        for 2D input. Passing ``axis=0`` raises ``NotImplementedError``.

    Returns
    -------
    y : np.ndarray, float32
        Softmax output, same shape as ``x``.
    """
    x = np.require(x, dtype=np.float32, requirements=["C_CONTIGUOUS"])

    if x.size == 0:
        return np.empty(x.shape, dtype=np.float32)

    # Handle 1D input by reshaping to (1, N)
    original_ndim = x.ndim
    if x.ndim not in (1, 2):
        raise NotImplementedError(
            f"softmax on {x.ndim}D input is not implemented (only 1D and 2D are supported)"
        )
    if x.ndim == 1:
        x = x.reshape(1, -1)

    # Axis validation: only the last dimension is supported.
    # Compute the positive axis index and verify it's the last dimension.
    pos_axis = axis if axis >= 0 else original_ndim + axis
    if pos_axis != original_ndim - 1:
        raise NotImplementedError(
            f"softmax along axis={axis} on {original_ndim}D input is not implemented "
            f"(only the last dimension, axis={original_ndim - 1}, is supported)"
        )

    M, N = x.shape

    # Compute row stride in elements (not bytes)
    stride_x = x.strides[0] // x.itemsize

    x_dev = gpu.to_device(x)
    y_dev = gpu.allocate(x.shape, np.float32)
    stride_y = y_dev.shape[1]  # row stride for output in elements

    # Adaptive block size: smallest power-of-2 >= N, capped at 1024 to avoid
    # wasting lanes for very small rows. Minimum 32 (warp size on NVIDIA).
    max_block = 1024
    min_block = 32
    BLOCK_SIZE = max(min_block, triton.next_power_of_2(min(N, max_block)))
    grid = (M,)  # one program per row

    _softmax_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(),
        M, N,
        stride_x, stride_y,
        BLOCK_SIZE,
        SINGLE_BLOCK=(N <= BLOCK_SIZE),
    )

    gpu.synchronize()
    y = gpu.to_host(y_dev)

    # Flatten back if input was 1D
    if original_ndim == 1:
        y = y.ravel()

    return y
