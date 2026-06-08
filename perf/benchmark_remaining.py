#!/usr/bin/env python3
"""Remaining benchmark cases: prefill-heavy, balanced, near-limit (gen ≤ 256)."""
import time, numpy as np, sys, os
sys.path.insert(0, '.')
os.environ["PYTHONUNBUFFERED"] = "1"
from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM

def make_weights(config):
    n, v, n_kv, n_h, nl, ffn = config.n_embd, config.vocab_size, config.n_kv_head, config.n_head, config.n_layer, config.n_ffn
    d_k = n // n_h
    w = {"model.embed_tokens.weight": np.random.randn(v, n).astype(np.float32) * 0.02,
         "model.norm.weight": np.random.randn(n).astype(np.float32) * 0.02}
    for i in range(nl):
        w.update({f"model.layers.{i}.input_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.post_attention_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.self_attn.q_proj.weight": np.random.randn(n_h*d_k, n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.self_attn.k_proj.weight": np.random.randn(n_kv*d_k, n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.self_attn.v_proj.weight": np.random.randn(n_kv*d_k, n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.self_attn.o_proj.weight": np.random.randn(n, n_h*d_k).astype(np.float32) * 0.02,
                  f"model.layers.{i}.mlp.gate_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.mlp.up_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
                  f"model.layers.{i}.mlp.down_proj.weight": np.random.randn(n, ffn).astype(np.float32) * 0.02})
    return w

def gen_full(model, tokens, n):
    t = tokens.copy()
    for _ in range(n):
        logits = model._forward_full(t)
        nt = int(np.argmax(logits[0, -1, :]))
        t = np.concatenate([t, np.array([[nt]], dtype=np.int32)], axis=1)
    return t

def gen_cache(model, tokens, n):
    return model.generate(tokens.copy(), max_new_tokens=n, temperature=0.0)

# Only remaining cases: prefill-heavy, balanced, near-limit
test_cases = [
    ("prefill-heavy", 256, 10),
    ("prefill-heavy", 512, 10),
    ("prefill-heavy", 1024, 10),
    ("prefill-heavy", 2048, 10),
    ("balanced", 128, 128),
    ("balanced", 256, 64),
    ("balanced", 512, 128),
    ("balanced", 1024, 256),
    ("near-limit", 4096, 10),
    ("near-limit", 2048, 100),
    ("near-limit", 512, 512),
]

config = SmolLM2Config(hidden_size=576, num_hidden_layers=30, num_attention_heads=9,
                        num_key_value_heads=3, intermediate_size=1536, vocab_size=49152,
                        max_position_embeddings=8192)

print(f"SmolLM2-135M (30×576, 9H, 3KV) | H100 | batch=1 | float32", flush=True)
print(f"{'case':<16} {'prompt':>6} {'gen':>6} {'total':>6} | "
      f"{'full(s)':<10} {'cache(s)':<10} {'speedup':<8} "
      f"{'pt_full(ms)':<10} {'pt_cache(ms)':<8} quality", flush=True)
print("-" * 95, flush=True)

for case_name, prompt_len, gen_len in test_cases:
    total_seq = prompt_len + gen_len
    if total_seq > config.max_position_embeddings:
        print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | SKIP", flush=True)
        continue

    np.random.seed(42)
    prompt = np.random.randint(0, config.vocab_size, (1, prompt_len)).astype(np.int32)
    w = make_weights(config)

    # Full recompute
    m = SmolLM2ForCausalLM(config, w)
    t0 = time.time()
    out_full = gen_full(m, prompt.copy(), gen_len)
    t_full = time.time() - t0
    del m

    # KV cache
    m = SmolLM2ForCausalLM(config, w)
    t0 = time.time()
    out_cache = gen_cache(m, prompt.copy(), gen_len)
    t_cache = time.time() - t0
    del m

    qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
    perfull = (t_full / gen_len) * 1000 if gen_len > 0 else 0.0
    percache = (t_cache / gen_len) * 1000 if gen_len > 0 else 0.0
    spd = t_full / t_cache if t_cache > 0 else float('inf')

    print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | "
          f"{t_full:<10.4f} {t_cache:<10.4f} {spd:<8.2f}x "
          f"{perfull:<10.1f} {percache:<10.1f} {qual}", flush=True)
    print("-" * 95, flush=True)
