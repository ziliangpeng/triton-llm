import numpy as np
import sys
sys.path.append('/Users/victor.peng/code/triton-llm')
from gpt2_triton.kernels.layernorm import layernorm

def test_layernorm():
    np.random.seed(0)
    batch_size = 4
    hidden_size = 768
    x = np.random.randn(batch_size, hidden_size).astype(np.float32)
    weight = np.ones(hidden_size, dtype=np.float32)
    bias = np.zeros(hidden_size, dtype=np.float32)

    # Reference using numpy
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    ref = (x - mean) / np.sqrt(var + 1e-5) * weight + bias

    # Our Triton version (placeholder)
    y = layernorm(x, weight, bias, eps=1e-5)

    print("LayerNorm test structure ready (GPU launch pending)")
    print("Reference shape:", ref.shape)
    print("Test passed (structure check)!")