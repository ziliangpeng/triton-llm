#!/usr/bin/env python3
"""Benchmark all 4 GPT-2 variants: side-by-side full recompute vs KV cache on H100.

Prints raw timings (total + per_token) for both modes, plus speedup and quality.
Uses TRITON_CACHE_DIR to avoid repeated compilation."""
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
    t = tokens.copy()
    for _ in range(n):
        logits = model._forward_full(t)
        nt = int(np.argmax(logits[0, -1, :]))
        t = np.concatenate([t, np.array([[nt]], dtype=np.int32)], axis=1)
    return t

configs = [
    ("Small",  GPT2Config(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257, n_positions=1024)),
    ("Medium", GPT2Config(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, n_positions=1024)),
    ("Large",  GPT2Config(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, n_positions=1024)),
    ("XL",     GPT2Config(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, n_positions=1024)),
]

# Warmup first (Small)
print("Warmup Small...", flush=True)
w = _weights(configs[0][1])
m = GPT2Model(configs[0][1], w)
_ = m.generate(np.array([[5,12]], dtype=np.int32), max_new_tokens=1, temperature=0.0)
del m, w
print("Warmup done.\n", flush=True)

for name, config in configs:
    print(f"\n{'='*70}")
    print(f"GPT-2 {name} ({config.n_layer}L×{config.n_embd}E)")
    print(f"{'='*70}")

    np.random.seed(42)
    prompt = np.random.randint(0, config.vocab_size, (1, 8)).astype(np.int32)
    gen_lens = [10, 30, 50]

    header = f"{'gen':>5} {'mode':<14} {'total(s)':<10} {'per_token(ms)':<14} {'speedup':<10} {'quality':<10}"
    sep = "-" * 65
    print(header)
    print(sep)

    for glen in gen_lens:
        w = _weights(config)

        m = GPT2Model(config, w)
        t0 = time.time()
        out_full = gen_full(m, prompt.copy(), glen)
        t_full = time.time() - t0
        pt_full = (t_full / glen) * 1000
        del m

        m = GPT2Model(config, w)
        t0 = time.time()
        out_cache = m.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
        t_cache = time.time() - t0
        pt_cache = (t_cache / glen) * 1000
        del m

        qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
        spd = t_full / t_cache if t_cache > 0 else float('inf')

        print(f"{glen:>5} {'full-recompute':<14} {t_full:<10.4f} {pt_full:<14.1f} {'-':<10} {'-':<10}")
        print(f"{glen:>5} {'kv-cache':<14} {t_cache:<10.4f} {pt_cache:<14.1f} {spd:<10.2f}x {qual:<10}")
        print(sep)

print("\nH100 | Triton 3.4.0 | CUDA 13.0 | batch=1 | float32 | random weights | prompt=8")
