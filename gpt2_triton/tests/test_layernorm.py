import torch
import sys
sys.path.append('/Users/victor.peng/code/triton-llm')
from gpt2_triton.kernels.layernorm import layernorm

def test_layernorm():
    torch.manual_seed(0)
    batch_size = 4
    hidden_size = 768
    x = torch.randn(batch_size, hidden_size, device='cuda')
    weight = torch.ones(hidden_size, device='cuda')
    bias = torch.zeros(hidden_size, device='cuda')

    # Reference using torch
    ref = torch.nn.functional.layer_norm(x, (hidden_size,), weight, bias, eps=1e-5)

    # Our Triton version
    y = layernorm(x, weight, bias, eps=1e-5)

    max_diff = (y - ref).abs().max().item()
    print(f"Max diff: {max_diff:.6e}")
    assert max_diff < 1e-3, "LayerNorm result too different from reference"
    print("LayerNorm test passed!")

if __name__ == "__main__":
    test_layernorm()
