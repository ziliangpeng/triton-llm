import triton
import triton.language as tl
import numpy as np
from .. import gpu

@triton.jit
def _layernorm_kernel(
    X,
    Y,
    W,
    B,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * N
    y_ptr = Y + row * N

    mean = 0.0
    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        mean += tl.sum(x)
    mean /= N

    var = 0.0
    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        var += tl.sum((x - mean) * (x - mean))
    var /= N
    rstd = 1.0 / tl.sqrt(var + eps)

    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        w = tl.load(W + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        b = tl.load(B + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + i + tl.arange(0, BLOCK_SIZE), y, mask=i + tl.arange(0, BLOCK_SIZE) < N)

def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
    """Runs LayerNorm on GPU using Triton. Returns numpy array."""
    x_dev = gpu.to_device(x)
    w_dev = gpu.to_device(weight)
    b_dev = gpu.to_device(bias)
    y_dev = gpu.allocate(x.shape, x.dtype)

    N = x.shape[-1]
    grid = (x.shape[0],)
    BLOCK_SIZE = 128 if N >= 128 else N

    _layernorm_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(),
        w_dev.data_ptr(), b_dev.data_ptr(),
        N, eps, BLOCK_SIZE
    )

    return gpu.to_host(y_dev)