"""
Triton SwiGLU Kernel (CUDA + HIP)

Element-wise SwiGLU activation = silu(x) * y, where:
    silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

Numerically stable silu via clamping x to [-20, 20] before exp
to prevent float32 overflow from exp(large_positive).

Reference: https://arxiv.org/abs/2002.05202 (Swish/SiLU) applied
as gated activation in Llama-architecture FFN blocks.
"""

import triton
import triton.language as tl
import numpy as np
from gpt2_triton import gpu


@triton.jit
def _swiglu_kernel(
    X,  # gate (raw device ptr)
    Y,  # up (raw device ptr)
    O,  # output (raw device ptr)
    N,  # number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise SwiGLU: O[i] = silu(X[i]) * Y[i].

    Casts raw int64 pointers to typed float32 pointers (Triton 3.x compat).
    Masks out-of-bounds lanes to 0.0.
    """
    # Cast raw pointers to typed float32 pointers (Triton 3.x compat)
    X = tl.cast(X, tl.pointer_type(tl.float32))
    Y = tl.cast(Y, tl.pointer_type(tl.float32))
    O = tl.cast(O, tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load gate and up.
    x = tl.load(X + offsets, mask=mask, other=0.0)
    y = tl.load(Y + offsets, mask=mask, other=0.0)

    # Numerically stable sigmoid via abs-based single-exp formulation.
    # For x >= 0: sigmoid = 1 / (1 + exp(-|x|))
    # For x < 0:  sigmoid = exp(-|x|) / (1 + exp(-|x|))
    # exp(-|x|) is always safe (input ≤ 0, output ≤ 1).
    exp_neg_abs = tl.exp(-tl.abs(x))
    one = 1.0
    sigmoid = tl.where(x >= 0, one / (one + exp_neg_abs), exp_neg_abs / (one + exp_neg_abs))
    silu = x * sigmoid

    result = silu * y
    tl.store(O + offsets, result, mask=mask)


def _next_pow2(n: int) -> int:
    """Smallest power of two >= n (>=1). Returns 1 for n <= 0."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Apply SwiGLU activation: silu(gate) * up (element-wise).

    Parameters
    ----------
    gate : np.ndarray, float32. Any shape.
    up : np.ndarray, float32. Same shape as ``gate``.

    Returns
    -------
    out : np.ndarray, same shape as inputs, float32.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate shape {gate.shape} does not match up shape {up.shape}"
        )

    orig_shape = gate.shape
    N = gate.size

    if N == 0:
        return np.empty(orig_shape, dtype=np.float32)

    # Ensure contiguous float32.
    gate = np.require(gate, dtype=np.float32, requirements=['C_CONTIGUOUS'])
    up = np.require(up, dtype=np.float32, requirements=['C_CONTIGUOUS'])

    # Transfer to device (flattened).
    gate_dev = gpu.to_device(gate.reshape(-1))
    up_dev = gpu.to_device(up.reshape(-1))
    out_dev = gpu.allocate((N,), np.float32)

    BLOCK_SIZE = min(_next_pow2(N), 1024)
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _swiglu_kernel[grid](
        gate_dev.data_ptr(),
        up_dev.data_ptr(),
        out_dev.data_ptr(),
        N,
        BLOCK_SIZE,
    )

    gpu.synchronize()
    return gpu.to_host(out_dev).reshape(orig_shape)
