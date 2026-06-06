#!/usr/bin/env python3
"""Quick benchmark: GPT-2 Small, full recompute vs KV cache."""
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

configs = [
    ("GPT-2 Small",  GPT2Config(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257, n_positions=1024)),
    ("GPT-2 Medium", GPT2Config(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, n_positions=1024)),
    ("GPT-2 Large",  GPT2Config(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, n_positions=1024)),
    ("GPT-2 XL",     GPT2Config(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, n_positions=1024)),
]

print("=" * 100)
print("GPT-2 Generation Benchmark: Full Recompute vs KV Cache")
print("=" * 100)

for name, config in configs:
    print(f"\n--- {name} (n_layer={config.n_layer}, n_embd={config.n_embd}) ---")
    prompt = np.random.randint(0, config.vocab_size, (1, 8)).astype(np.int32)
    weights = _weights(config)

    for glen in [1, 10, 30]:
        # Full
        m = GPT2Model(config, weights)
        t0 = time.time()
        out_full = generate_full(m, prompt.copy(), glen)
        t_full = time.time() - t0
        del m

        # Cache
        m = GPT2Model(config, weights)
        t0 = time.time()
        out_cache = m.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
        t_cache = time.time() - t0
        del m

        qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
        spd = t_full / t_cache if t_cache > 0 else float('inf')
        print(f"  prompt=8  gen={glen:<3}  full={t_full:<8.4f}s  cache={t_cache:<8.4f}s  {spd:<5.1f}x  quality={qual}")
