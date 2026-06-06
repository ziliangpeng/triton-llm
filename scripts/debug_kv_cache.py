#!/usr/bin/env python3
"""Debug: trace where cached and full-recompute diverge."""
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

# Use GPT-2 Small config
config = GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, n_positions=1024)
prompt = np.random.randint(0, config.vocab_size, (1, 8)).astype(np.int32)
weights = _weights(config)

# Step 1: prefill comparison
m1 = GPT2Model(config, weights)
m2 = GPT2Model(config, weights)

# Full (no cache) prefill
full_logits = m1._forward_full(prompt)

# Cached prefill
m2._init_cache()
cached_logits = m2._forward_cached(prompt)

prefill_max_diff = float(np.abs(full_logits - cached_logits).max())
print(f"Prefill max_diff: {prefill_max_diff:.6e}")
print(f"Prefill logits match: {np.allclose(full_logits, cached_logits, atol=1e-4)}")
print(f"Argmax at last pos - full: {int(np.argmax(full_logits[0,-1,:]))}, cached: {int(np.argmax(cached_logits[0,-1,:]))}")

del m1, m2

# Step 2: single decode step
m_full = GPT2Model(config, weights)
m_cache = GPT2Model(config, weights)

# Full recompute for 2 steps
tokens = prompt.copy()
step_logits_full = []
for s in range(2):
    logits = m_full._forward_full(tokens)
    step_logits_full.append(logits[0, -1, :].copy())
    nt = int(np.argmax(logits[0, -1, :]))
    tokens = np.concatenate([tokens, np.array([[nt]], dtype=np.int32)], axis=1)
out_full = tokens

# Cached for 2 steps
m_cache._init_cache()
logits = m_cache._forward_cached(prompt)  # prefill
tokens = prompt.copy()
step_logits_cache = [logits[0, -1, :].copy()]
nt = int(np.argmax(logits[0, -1, :]))
tokens = np.concatenate([tokens, np.array([[nt]], dtype=np.int32)], axis=1)

# Step 2: decode
new_token = np.array([[nt]], dtype=np.int32)
logits = m_cache._forward_cached(new_token)
step_logits_cache.append(logits[0, -1, :].copy())

print(f"\nStep 0 logits max_diff: {float(np.abs(step_logits_full[0] - step_logits_cache[0]).max()):.6e}")
print(f"Step 0 tokens: {int(np.argmax(step_logits_full[0]))} vs {int(np.argmax(step_logits_cache[0]))}")

# Check the full recompute at step 1
# Full recompute processes tokens 0..8 (8 prompt + 1 generated)
full_step1 = step_logits_full[1]  # from processing all 9 tokens
cache_step1 = step_logits_cache[1]  # from processing just token 8 with cached K/V

print(f"\nStep 1 logits max_diff: {float(np.abs(full_step1 - cache_step1).max()):.6e}")
print(f"Step 1 logits match atol=1e-4: {np.allclose(full_step1, cache_step1, atol=1e-4)}")
print(f"Step 1 tokens: {int(np.argmax(full_step1))} vs {int(np.argmax(cache_step1))}")

# Deeper: compare the full forward for seq=9 vs cached forward for single token
# At the first full forward for 9 tokens, position 8 is the last token
# At the cached forward, we process token 8 alone

print(f"\n--- Full forward 9-token: last position logits[0,-1,:8] ---")
print(full_step1[:8])
print(f"--- Cached decode single-token: logits[0,0,:8] ---")
print(cache_step1[:8])

del m_full, m_cache
