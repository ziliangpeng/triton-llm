#!/usr/bin/env python3
"""Side-by-side benchmark: full recompute vs KV cache for SmolLM2.

Measures speedup across varied prompt/gen lengths, verifies output quality
(output tokens must match between full and cached paths).

Usage:
    srun --gres=gpu:1 bash -c 'PYTHONPATH=. python perf/benchmark_smollm2_kv.py'
"""
import time
import numpy as np
import sys
sys.path.insert(0, '.')

from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM


def make_weights(config):
    n = config.n_embd
    v = config.vocab_size
    n_kv = config.n_kv_head
    n_h = config.n_head
    d_k = n // n_h
    nl = config.n_layer
    ffn = config.n_ffn
    w = {
        "model.embed_tokens.weight": np.random.randn(v, n).astype(np.float32) * 0.02,
        "model.norm.weight": np.random.randn(n).astype(np.float32) * 0.02,
    }
    for i in range(nl):
        w.update({
            f"model.layers.{i}.input_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
            f"model.layers.{i}.post_attention_layernorm.weight": np.random.randn(n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.q_proj.weight": np.random.randn(n_h * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.k_proj.weight": np.random.randn(n_kv * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.v_proj.weight": np.random.randn(n_kv * d_k, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.self_attn.o_proj.weight": np.random.randn(n, n_h * d_k).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.gate_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.up_proj.weight": np.random.randn(ffn, n).astype(np.float32) * 0.02,
            f"model.layers.{i}.mlp.down_proj.weight": np.random.randn(n, ffn).astype(np.float32) * 0.02,
        })
    return w


def gen_full(model, tokens, n):
    """Greedy generate using full recompute each step (_forward_full)."""
    t = tokens.copy()
    for _ in range(n):
        logits = model._forward_full(t)
        nt = int(np.argmax(logits[0, -1, :]))
        t = np.concatenate([t, np.array([[nt]], dtype=np.int32)], axis=1)
    return t


def gen_cache(model, tokens, n):
    """Greedy generate using KV cache (model.generate)."""
    return model.generate(tokens.copy(), max_new_tokens=n, temperature=0.0)


# Test configurations: (name, prompt_len, gen_len)
# SmolLM2-135M: max_position_embeddings=8192
# Covers: GEMM-dominated, attention-growing, O(T²) regime
test_cases = [
    # Tiny — GEMM dominated, KV cache benefit should be minimal
    ("tiny",        8,    10),
    # Baseline short
    ("baseline",    8,    50),
    ("baseline",    8,   100),
    # Decode-heavy: long generation, short prompt
    ("decode-heavy", 8,   200),
    ("decode-heavy", 8,   500),
    ("decode-heavy", 8,  1000),
    # Prefill-heavy: long prompt, short generation
    ("prefill-heavy", 256,  10),
    ("prefill-heavy", 512,  10),
    ("prefill-heavy", 1024, 10),
    ("prefill-heavy", 2048, 10),
    # Balanced: both prompt and gen are substantial
    ("balanced",    128,  128),
    ("balanced",    256,   64),
    ("balanced",    512,  128),
    ("balanced",   1024,  256),
    # Near limit: max context (for SmolLM2 that's 8192)
    ("near-limit",  4096,   10),
    ("near-limit",  2048,  100),
    ("near-limit",  512,  512),
]

# Long prompted quality checks compare each decode step's token identity
# and aggregate as fraction of matches (not just final sequence identity),
# because a single divergence early in sampling cascades.
# Greedy (temperature=0) avoids this — outputs MUST be identical.


def main():
    config = SmolLM2Config(
        hidden_size=576, num_hidden_layers=30,
        num_attention_heads=9, num_key_value_heads=3,
        intermediate_size=1536, vocab_size=49152,
        max_position_embeddings=8192,
    )

    # Warmup
    print("Warmup...", flush=True)
    w = make_weights(config)
    m = SmolLM2ForCausalLM(config, w)
    _ = m.generate(np.array([[5, 12]], dtype=np.int32), max_new_tokens=2, temperature=0.0)
    del m, w
    print("Warmup done.\n", flush=True)

    print("=" * 115, flush=True)
    print("SmolLM2-135M (30×576, 9H, 3KV) | H100 | Triton 3.4.0 | CUDA 13.0 | batch=1 | float32", flush=True)
    print("=" * 115, flush=True)

    header = (f"{'case':<16} {'prompt':>6} {'gen':>6} {'total':>6} | "
              f"{'full(s)':<10} {'cache(s)':<10} {'speedup':<8} "
              f"{'pt_full(ms)':<10} {'pt_cache(ms)':<10} {'quality':<8} "
              f"{'cache_len':>9}")
    print(header, flush=True)
    print("-" * 115, flush=True)

    for case_name, prompt_len, gen_len in test_cases:
        total_seq = prompt_len + gen_len
        if total_seq > config.max_position_embeddings:
            print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | SKIP (OOB)", flush=True)
            continue

        np.random.seed(42)
        prompt = np.random.randint(0, config.vocab_size, (1, prompt_len)).astype(np.int32)
        w = make_weights(config)

        # --- Full recompute ---
        m = SmolLM2ForCausalLM(config, w)
        t0 = time.time()
        out_full = gen_full(m, prompt.copy(), gen_len)
        t_full = time.time() - t0
        del m

        # --- KV cache ---
        m = SmolLM2ForCausalLM(config, w)
        t0 = time.time()
        out_cache = gen_cache(m, prompt.copy(), gen_len)
        t_cache = time.time() - t0
        final_cache_len = m._cache_len if hasattr(m, '_cache_len') else 'N/A'
        del m

        # Quality: greedy decode must produce identical tokens at every step
        qual = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"

        # Per-token timing (total time / num_decode_steps = gen_len)
        perfull = (t_full / gen_len) * 1000 if gen_len > 0 else 0.0
        percache = (t_cache / gen_len) * 1000 if gen_len > 0 else 0.0
        spd = t_full / t_cache if t_cache > 0 else float('inf')

        print(f"{case_name:<16} {prompt_len:>6} {gen_len:>6} {total_seq:>6} | "
              f"{t_full:<10.4f} {t_cache:<10.4f} {spd:<8.2f}x "
              f"{perfull:<10.1f} {percache:<10.1f} {qual:<8} "
              f"{final_cache_len:>9}", flush=True)
        print("-" * 115, flush=True)

    print(flush=True)
    print("Notes:", flush=True)
    print("  'full'  = full recompute (O(T²) each decode step)", flush=True)
    print("  'cache' = KV cache incremental decode (O(T) each decode step)", flush=True)
    print("  quality = token identity check between full and cached (greedy)", flush=True)
    print("  pt_*   = per-decode-step milliseconds", flush=True)
    print("  cache_len = KV cache position count after generation", flush=True)


if __name__ == "__main__":
    main()
