import triton
import triton.language as tl
import numpy as np
from .. import gpu

@triton.jit
def _add_kernel(
    X, Y, Z,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(Z + offs, x + y, mask=mask)

def add(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape
    z_dev = gpu.allocate(x.shape, x.dtype)
    x_dev = gpu.to_device(x)
    y_dev = gpu.to_device(y)

    N = x.size
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _add_kernel[grid](
        x_dev.data_ptr(), y_dev.data_ptr(), z_dev.data_ptr(),
        N, BLOCK_SIZE
    )
    return gpu.to_host(z_dev)