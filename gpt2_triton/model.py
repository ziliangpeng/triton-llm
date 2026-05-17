"""
Minimal GPT-2 model using Triton kernels (no PyTorch).

This is a work-in-progress implementation focused on getting
a working forward pass using our custom Triton kernels.
"""

import numpy as np
from .kernels.layernorm import layernorm
from .kernels.gelu import gelu
from .kernels.gemm import gemm


class Config:
    def __init__(self, vocab_size=1000, n_embd=128, n_layer=2, n_head=4):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head


class LayerNorm:
    def __init__(self, ndim):
        self.weight = np.ones(ndim, dtype=np.float32)
        self.bias = np.zeros(ndim, dtype=np.float32)

    def __call__(self, x):
        return layernorm(x, self.weight, self.bias)


class Linear:
    def __init__(self, in_features, out_features):
        scale = 1.0 / np.sqrt(in_features)
        self.weight = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.bias = np.zeros(out_features, dtype=np.float32)

    def __call__(self, x):
        # Use our Triton GEMM
        return gemm(x, self.weight) + self.bias


class MLP:
    def __init__(self, config):
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        return x


class Attention:
    def __init__(self, config):
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head

    def __call__(self, x):
        # Simplified attention (numpy for now)
        B, T, C = x.shape
        qkv = self.c_attn(x.reshape(B*T, C)).reshape(B, T, 3*C)
        q, k, v = np.split(qkv, 3, axis=2)

        # Naive attention
        att = (q @ k.transpose(0, 2, 1)) / np.sqrt(k.shape[-1])
        att = np.tril(att)
        att = np.exp(att - att.max(axis=-1, keepdims=True))
        att = att / att.sum(axis=-1, keepdims=True)
        y = att @ v
        return self.c_proj(y.reshape(B*T, C)).reshape(B, T, C)


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
        self.wte = np.random.randn(config.vocab_size, config.n_embd).astype(np.float32) * 0.02
        self.wpe = np.random.randn(1024, config.n_embd).astype(np.float32) * 0.01
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
        logits = gemm(x.reshape(B*T, -1), self.wte.T).reshape(B, T, -1)
        return logits


if __name__ == "__main__":
    config = Config(vocab_size=1000, n_embd=128, n_layer=2)
    model = GPT2(config)

    # Simple test input
    idx = np.array([[1, 5, 23, 99, 42]], dtype=np.int32)
    logits = model(idx)

    print(f"Input shape : {idx.shape}")
    print(f"Logits shape: {logits.shape}")
    print("Minimal GPT-2 forward pass completed successfully!")