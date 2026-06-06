#!/usr/bin/env python3
"""Benchmark: GPT-2 Small, full recompute vs KV cache on H100."""
import time, numpy as np, sys
sys.path.insert(0, '.')
from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model

def _weights(c):
    n, v, nl = c.n_embd, c.vocab_size, c.n_layer
    w = {"wte.weight": np.random.randn(v,n).astype(np.float32),
         "wpe.weight": np.random.randn(c.n_positions,n).astype(np.float32),
         "ln_f.weight": np.random.randn(n).astype(np.float32),
         "ln_f.bias": np.random.randn(n).astype(np.float32)}
    for i in range(nl):
        w.update({f"h.{i}.attn.c_attn.weight": np.random.randn(n,3*n).astype(np.float32),
                  f"h.{i}.attn.c_attn.bias": np.random.randn(3*n).astype(np.float32),
                  f"h.{i}.attn.c_proj.weight": np.random.randn(n,n).astype(np.float32),
                  f"h.{i}.attn.c_proj.bias": np.random.randn(n).astype(np.float32),
                  f"h.{i}.mlp.c_fc.weight": np.random.randn(n,4*n).astype(np.float32),
                  f"h.{i}.mlp.c_fc.bias": np.random.randn(4*n).astype(np.float32),
                  f"h.{i}.mlp.c_proj.weight": np.random.randn(4*n,n).astype(np.float32),
                  f"h.{i}.mlp.c_proj.bias": np.random.randn(n).astype(np.float32),
                  f"h.{i}.ln_1.weight": np.random.randn(n).astype(np.float32),
                  f"h.{i}.ln_1.bias": np.random.randn(n).astype(np.float32),
                  f"h.{i}.ln_2.weight": np.random.randn(n).astype(np.float32),
                  f"h.{i}.ln_2.bias": np.random.randn(n).astype(np.float32)})
    return w

def gen_full(model, tokens, n):
    """Full recompute generation (greedy)."""
    t = tokens.copy()
    for _ in range(n):
        logits = model._forward_full(t)
        nt = int(np.argmax(logits[0, -1, :]))
        t = np.concatenate([t, np.array([[nt]], dtype=np.int32)], axis=1)
    return t

np.random.seed(42)
config = GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, n_positions=1024)

# Warmup / Triton compile
print("Warmup (compiling Triton kernels)...", flush=True)
w = _weights(config)
m = GPT2Model(config, w)
_ = m.generate(np.array([[5, 12]], dtype=np.int32), max_new_tokens=1, temperature=0.0)
_ = gen_full(m, np.array([[5, 12]], dtype=np.int32), 1)
del m
print("Warmup done.\n")

print("=" * 90)
print("GPT-2 Small (12×768): Full Recompute vs KV Cache (H100)")
print("=" * 90)
print(f"{'prompt':<8} {'gen':<6} {'full(s)':<12} {'cache(s)':<12} {'speedup':<10} {'quality':<10}")
print("-" * 90)

for plen in [8, 32]:
    prompt = np.random.randint(0, config.vocab_size, (1, plen)).astype(np.int32)
    for glen in [10, 30, 50]:
        w = _weights(config)

        # Full recompute
        m = GPT2Model(config, w)
        t0 = time.time()
        out_full = gen_full(m, prompt.copy(), glen)
        t_full = time.time() - t0
        del m

        # KV cache
        m = GPT2Model(config, w)
        t0 = time.time()
        out_cache = m.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
        t_cache = time.time() - t0
        del m

        qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
        spd = t_full / t_cache if t_cache > 0 else float('inf')
        print(f"{plen:<8} {glen:<6} {t_full:<12.4f} {t_cache:<12.4f} {spd:<10.2f}x {qual:<10}")

print("-" * 90)
print("H100 | Triton 3.4.0 | CUDA 13.0 | batch=1 | float32 | random weights")
print("Quality = cached greedy output matches full-recompute greedy output")
