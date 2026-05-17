import torch
import sys
sys.path.append('/Users/victor.peng/code/triton-llm')
from gpt2_triton.kernels.gelu import gelu

def test_gelu():
    torch.manual_seed(0)
    x = torch.randn(1024, device='cuda')

    ref = torch.nn.functional.gelu(x, approximate='tanh')

    y = gelu(x)
    max_diff = (y - ref).abs().max().item()
    print(f"Max diff: {max_diff:.6e}")
    assert max_diff < 1e-4, "GELU result too different"
    print("GELU test passed!")

if __name__ == "__main__":
    test_gelu()
