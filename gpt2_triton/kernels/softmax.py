"""
Triton Softmax Kernel (CUDA + HIP) — numerically stable two-pass softmax.

Uses the standard two-pass reduction pattern:
  Pass 1: row_max = max(x_row)
  Pass 2: row_sum = sum(exp(x_row - row_max))
  Pass 3: y = exp(x - row_max) / row_sum

Each Triton program handles one row of the input matrix.
"""

import triton
import triton.language as tl
import numpy as np
from .. import gpu


@triton.jit
def _softmax_kernel(
    X,
    Y,
    M,
    N,
    stride_x,
    stride_y,
    BLOCK_SIZE: tl.constexpr,
):
    """Numerically stable two-pass softmax on a single matrix row.

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
    """
    row_idx = tl.program_id(0)
    # Cast raw pointers to typed float32 pointers before arithmetic
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    x_ptr = X + row_idx * stride_x
    y_ptr = Y + row_idx * stride_y

    offs = tl.arange(0, BLOCK_SIZE)

    # --- Pass 1: compute row-wise maximum ---
    row_max = -float("inf")
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
        block_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # --- Pass 2: compute sum of exp(x - row_max) ---
    row_sum = 0.0
    for start in range(0, N, BLOCK_SIZE):
        cols = start + offs
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=-float("inf"))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # --- Pass 3: compute softmax values ---
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
    )

    gpu.synchronize()
    y = gpu.to_host(y_dev)

    # Flatten back if input was 1D
    if original_ndim == 1:
        y = y.ravel()

    return y
