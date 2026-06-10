"""Triton element-wise Add Kernel (Z = X + Y) — CUDA + HIP.

Adds two numeric arrays element-wise (auto-converted to float32) using the
  to_device -> launch -> synchronize -> to_host
"""

import triton
import triton.language as tl
import numpy as np
from triton_llm import gpu


@triton.jit
def _add_kernel(
    X,
    Y,
    Z,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition Z = X + Y on flattened 1-D arrays.

    Each program handles ``BLOCK_SIZE`` elements. Out-of-bounds elements
    (when ``N`` is not a multiple of ``BLOCK_SIZE``) are masked out.

    Parameters
    ----------
    X : int
        Raw int64 device pointer for the first operand (cast to ``float32`` inside).
    Y : int
        Raw int64 device pointer for the second operand (cast to ``float32`` inside).
    Z : int
        Raw int64 device pointer for the output (cast to ``float32`` inside).
    N : int
        Total number of elements (product of all dimensions).
    BLOCK_SIZE : tl.constexpr
        Number of elements processed per program.
    """
    # Cast raw int64 pointers to typed float32 pointers (Triton 3.x compatibility).
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    Z = tl.cast(Z, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)
    tl.store(Z + offsets, x + y, mask=mask)


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Element-wise addition of two arrays (C = A + B).

    Both inputs are converted to C-contiguous float32 before copying to
    the GPU. Output is returned as a NumPy float32 array with the same
    shape as the inputs.

    Parameters
    ----------
    x : np.ndarray
        First operand. Any shape and numeric dtype is accepted; it will
        be converted to float32 internally.
    y : np.ndarray
        Second operand. Must have the same shape as ``x``.

    Returns
    -------
    z : np.ndarray, float32
        Element-wise sum, same shape as the inputs.

    Raises
    ------
    AssertionError
        If ``x.shape != y.shape``.
    """
    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"

    if x.size == 0:
        return np.empty(x.shape, dtype=np.float32)

    x_dev = gpu.to_device(np.require(x, dtype=np.float32, requirements=['C_CONTIGUOUS']))
    y_dev = gpu.to_device(np.require(y, dtype=np.float32, requirements=['C_CONTIGUOUS']))
    z_dev = gpu.allocate(x.shape, np.float32)
    N = x.size
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _add_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(), z_dev.data_ptr(),
        N, BLOCK_SIZE,
    )

    gpu.synchronize()
    return gpu.to_host(z_dev)


def add_device(
    x_dev: "gpu.DeviceTensor",
    y_dev: "gpu.DeviceTensor",
    out_dev: "gpu.DeviceTensor | None" = None,
) -> "gpu.DeviceTensor":
    """GPU-resident element-wise addition. No sync, no host copies.

    Parameters
    ----------
    x_dev : DeviceTensor, any shape, float32
        First operand on GPU.
    y_dev : DeviceTensor, same shape as x_dev, float32
        Second operand on GPU.
    out_dev : DeviceTensor, optional
        Pre-allocated output. Auto-allocated if None.

    Returns
    -------
    out_dev : DeviceTensor, same shape as inputs, float32
    """
    assert x_dev.shape == y_dev.shape, f"Shape mismatch: {x_dev.shape} vs {y_dev.shape}"
    if out_dev is not None:
        assert out_dev.shape == x_dev.shape, f"out_dev shape mismatch: expected {x_dev.shape}, got {out_dev.shape}"

    N = int(np.prod(x_dev.shape))
    if N == 0:
        if out_dev is None:
            out_dev = gpu.allocate(x_dev.shape, np.float32)
        return out_dev

    if out_dev is None:
        out_dev = gpu.allocate(x_dev.shape, np.float32)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _add_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(), out_dev.data_ptr(),
        N, BLOCK_SIZE,
    )

    return out_dev
