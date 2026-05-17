import torch
import sys
sys.path.append('/Users/victor.peng/code/triton-llm')
from gpt2_triton.kernels.gemm import gemm

def test_gemm():
    torch.manual_seed(0)
    M, K, N = 128, 256, 64
    a = torch.randn(M, K, device='cuda', dtype=torch.float32)
    b = torch.randn(K, N, device='cuda', dtype=torch.float32)

    ref = torch.matmul(a, b)
    c = gemm(a, b)

    max_diff = (c - ref).abs().max().item()
    print(f"Max diff: {max_diff:.6e}")
    assert max_diff < 1e-2, "GEMM result too different from reference"
    print("GEMM test passed!")

if __name__ == "__main__":
    test_gemm()