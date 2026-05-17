import triton
import triton.language as tl
import torch

@triton.jit
def _gelu_kernel(
    X,
    Y,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(X + offsets, mask=mask)
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    y = 0.5 * x * (1.0 + tl.tanh(sqrt_2_over_pi * (x + 0.044715 * x * x * x)))
    tl.store(Y + offsets, y, mask=mask)

def gelu(x: torch.Tensor):
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _gelu_kernel[grid](x, y, N, BLOCK_SIZE)
    return y
