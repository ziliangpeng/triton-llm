#!/usr/bin/env python3
"""Benchmark a single GPT-2 variant. Usage: python script.py small|medium|large|xl"""
import time, numpy as np, sys, os
sys.path.insert(0, '.')
from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model

variants = {
    "small":  GPT2Config(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257, n_positions=1024),
    "medium": GPT2Config(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, n_positions=1024),
    "large":  GPT2Config(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, n_positions=1024),
    "xl":     GPT2Config(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, n_positions=1024),
}

name = sys.argv[1].lower()
config = variants[name]

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
    t = tokens.copy()
    for _ in range(n):
        logits = model._forward_full(t)
        nt = int(np.argmax(logits[0, -1, :]))
        t = np.concatenate([t, np.array([[nt]], dtype=np.int32)], axis=1)
    return t

np.random.seed(42)
prompt = np.random.randint(0, config.vocab_size, (1, 8)).astype(np.int32)

print(f"GPT-2 {name.title()} ({config.n_layer}L×{config.n_embd}E)")
print(f"{'prompt':<8} {'gen':<6} {'full(s)':<12} {'cache(s)':<12} {'speedup':<10} {'quality':<10}")
print("-" * 60)

for glen in [10, 30, 50]:
    w = _weights(config)
    
    m = GPT2Model(config, w)
    t0 = time.time()
    out_full = gen_full(m, prompt.copy(), glen)
    t_full = time.time() - t0
    del m
    
    m = GPT2Model(config, w)
    t0 = time.time()
    out_cache = m.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
    t_cache = time.time() - t0
    del m
    
    qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
    spd = t_full / t_cache if t_cache > 0 else float('inf')
    print(f"{8:<8} {glen:<6} {t_full:<12.4f} {t_cache:<12.4f} {spd:<10.2f}x {qual:<10}")

print(f"\n{name} done.")
