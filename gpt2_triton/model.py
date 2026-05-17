"""
Pure Python + Triton GPT-2 model (no PyTorch)

This module provides a from-scratch GPT-2 implementation.
Currently uses numpy for CPU reference.
Triton kernels will be plugged in for GPU acceleration.
"""

import numpy as np
from .kernels.layernorm import layernorm as triton_layernorm
from .kernels.gelu import gelu as triton_gelu
from .kernels.gemm import gemm as triton_gemm

class GPT2Config:
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = 1024

class LayerNorm:
    def __init__(self, ndim, eps=1e-5):
        self.weight = np.ones(ndim, dtype=np.float32)
        self.bias = np.zeros(ndim, dtype=np.float32)
        self.eps = eps

    def __call__(self, x):
        # Use numpy reference for now
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps) * self.weight + self.bias

class Linear:
    def __init__(self, in_features, out_features):
        self.weight = np.random.randn(in_features, out_features).astype(np.float32) / np.sqrt(in_features)
        self.bias = np.zeros(out_features, dtype=np.float32)

    def __call__(self, x):
        return x @ self.weight + self.bias

class MLP:
    def __init__(self, config):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = triton_gelu(x) if hasattr(triton_gelu, '__call__') else 0.5 * x * (1 + np.tanh(0.79788456 * (x + 0.044715 * x**3)))
        x = self.c_proj(x)
        return x

class Attention:
    def __init__(self, config):
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = np.split(qkv, 3, axis=2)
        # Simple attention (for illustration)
        att = (q @ k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1])
        att = np.tril(att)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        y = att @ v
        y = self.c_proj(y)
        return y

class Block:
    def __init__(self, config):
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2:
    def __init__(self, config):
        self.config = config
        self.wte = np.random.randn(config.vocab_size, config.n_embd).astype(np.float32)
        self.wpe = np.random.randn(config.block_size, config.n_embd).astype(np.float32)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd)

    def __call__(self, idx):
        B, T = idx.shape
        tok_emb = self.wte[idx]
        pos_emb = self.wpe[:T]
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = x @ self.wte.T   # tie weights
        return logits

# Example usage (CPU reference)
if __name__ == "__main__":
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128)  # small for testing
    model = GPT2(config)
    idx = np.array([[1, 2, 3]], dtype=np.int32)
    logits = model(idx)
    print("Logits shape:", logits.shape)
    print("Basic GPT-2 skeleton works (numpy reference)")