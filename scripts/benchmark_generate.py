#!/usr/bin/env python3
"""Quick benchmark: GPT-2 Small, full vs KV cache."""
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

def generate_full(model, token_ids, max_new_tokens):
    tokens = token_ids.copy()
    for _ in range(max_new_tokens):
        logits = model._forward_full(tokens)
        next_token = int(np.argmax(logits[0, -1, :]))
        tokens = np.concatenate([tokens, np.array([[next_token]], dtype=np.int32)], axis=1)
    return tokens

np.random.seed(42)

config = GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, n_positions=1024)

print("=" * 80)
print("GPT-2 Small: Full Recompute vs KV Cache")
print("=" * 80)
print(f"\n{'prompt':<8} {'gen':<6} {'full(s)':<12} {'cache(s)':<12} {'speedup':<10} {'quality':<10}")
print("-" * 80)

prompt = np.random.randint(0, config.vocab_size, (1, 8)).astype(np.int32)

for glen in [1, 10, 30, 50]:
    weights = _weights(config)

    # Full
    m = GPT2Model(config, weights)
    t0 = time.time()
    logits = m._forward_full(prompt)
    tokens = prompt.copy()
    for _ in range(glen):
        logits = m._forward_full(tokens)
        nt = int(np.argmax(logits[0, -1, :]))
        tokens = np.concatenate([tokens, np.array([[nt]], dtype=np.int32)], axis=1)
    t_full = time.time() - t0
    out_full = tokens
    del m

    # Cache
    m = GPT2Model(config, weights)
    t0 = time.time()
    out_cache = m.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
    t_cache = time.time() - t0
    del m

    qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
    spd = t_full / t_cache if t_cache > 0 else float('inf')
    print(f"8        {glen:<6} {t_full:<12.4f} {t_cache:<12.4f} {spd:<10.2f}x {qual:<10}")

print("-" * 80)
print("H100 | GPT-2 Small (12-layer, 768-wide) | random weights")
