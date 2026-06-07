#!/usr/bin/env python3
"""Long-sequence benchmark: full recompute vs KV cache at varied prompt/gen lengths.

Tests how KV cache speedup changes as sequence length grows — the O(T²) vs O(T)
difference should become visible at longer sequences.

Usage:
    srun --gres=gpu:1 bash -c 'PYTHONPATH=. python perf/benchmark_long_seq.py'
"""
import time, numpy as np, sys
sys.path.insert(0, '.')
from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model

CONFIG = GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, n_positions=1024)

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

# Test configurations: (name, prompt_len, gen_len)
test_cases = [
    ("baseline",        8,   10),
    ("baseline",        8,   50),
    ("decode-heavy",    8,   100),
    ("decode-heavy",    8,   200),
    ("decode-heavy",    8,   500),
    ("prefill-heavy",   256, 10),
    ("prefill-heavy",   512, 10),
    ("balanced",        128, 128),
    ("balanced",        256, 64),
    ("near-limit",      510, 10),
    ("near-limit",      254, 100),
]

# Warmup
print("Warmup...", flush=True)
w = _weights(CONFIG)
m = GPT2Model(CONFIG, w)
_ = m.generate(np.array([[5,12]], dtype=np.int32), max_new_tokens=1, temperature=0.0)
del m, w
print("Warmup done.\n", flush=True)

print("=" * 95)
print("GPT-2 Small (12x768) | H100 | Triton 3.4.0 | CUDA 13.0 | batch=1 | float32")
print("=" * 95)

header = (f"{'case':<16} {'prompt':>6} {'gen':>6} {'total':>6} | "
          f"{'full(s)':<10} {'cache(s)':<10} {'speedup':<8} "
          f"{'pt_full(ms)':<10} {'pt_cache(ms)':<10} {'quality':<8}")
print(header)
print("-" * 95)

for case_name, prompt_len, gen_len in test_cases:
    total_seq = prompt_len + gen_len
    if total_seq > CONFIG.n_positions:
        print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | SKIP")
        continue

    np.random.seed(42)
    prompt = np.random.randint(0, CONFIG.vocab_size, (1, prompt_len)).astype(np.int32)
    w = _weights(CONFIG)

    m = GPT2Model(CONFIG, w)
    t0 = time.time()
    out_full = gen_full(m, prompt.copy(), gen_len)
    t_full = time.time() - t0
    del m

    m = GPT2Model(CONFIG, w)
    t0 = time.time()
    out_cache = m.generate(prompt.copy(), max_new_tokens=gen_len, temperature=0.0)
    t_cache = time.time() - t0
    del m

    qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
    spd = t_full / t_cache if t_cache > 0 else float('inf')

    print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | "
          f"{t_full:<10.4f} {t_cache:<10.4f} {spd:<8.2f}x "
          f"{(t_full/gen_len)*1000:<10.1f} {(t_cache/gen_len)*1000:<10.1f} {qual:<8}")
    print("-" * 95)

print("\nNote: 'full' = full recompute each step. 'cache' = KV cache incremental decode.")
print("Speedup > 1 = KV cache faster. Higher speedup at longer sequences confirms O(T²) vs O(T) benefit.")
