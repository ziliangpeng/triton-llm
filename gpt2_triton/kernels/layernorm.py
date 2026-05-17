import triton
import triton.language as tl
import numpy as np

@triton.jit
def _layernorm_kernel(
    X,  # input pointer
    Y,  # output pointer
    W,  # weight (gamma)
    B,  # bias (beta)
    N,  # hidden size
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * N
    y_ptr = Y + row * N

    # Compute mean
    mean = 0.0
    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        mean += tl.sum(x)
    mean /= N

    # Compute variance
    var = 0.0
    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        var += tl.sum((x - mean) * (x - mean))
    var /= N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply affine
    for i in range(0, N, BLOCK_SIZE):
        x = tl.load(x_ptr + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        w = tl.load(W + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        b = tl.load(B + i + tl.arange(0, BLOCK_SIZE), mask=i + tl.arange(0, BLOCK_SIZE) < N, other=0.0)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + i + tl.arange(0, BLOCK_SIZE), y, mask=i + tl.arange(0, BLOCK_SIZE) < N)

def layernorm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5):
    # Note: This wrapper assumes x, weight, bias are already on GPU device memory
    # For full no-torch implementation, a CUDA memory allocator (cupy or custom) is needed
    # For now, this is a placeholder that will be connected to a GPU backend
    N = x.shape[-1]
    y = np.empty_like(x)
    grid = (x.shape[0],)
    BLOCK_SIZE = 128 if N >= 128 else N
    _layernorm_kernel[grid](
        x, y, weight, bias, N, eps, BLOCK_SIZE
    )
    return y