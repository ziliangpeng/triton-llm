#!/usr/bin/env python3
"""Benchmark: full-recompute vs KV-cached generation.

Focused on GPT-2 Small with key gen lengths.
Tests quality equivalence and measures speedup.
"""

import time
import numpy as np
import sys
sys.path.insert(0, '.')

from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model


def _make_random_weights(config):
    n = config.n_embd
    v = config.vocab_size
    n_layer = config.n_layer
    weights = {
        "wte.weight": np.random.randn(v, n).astype(np.float32),
        "wpe.weight": np.random.randn(config.n_positions, n).astype(np.float32),
        "ln_f.weight": np.random.randn(n).astype(np.float32),
        "ln_f.bias": np.random.randn(n).astype(np.float32),
    }
    for i in range(n_layer):
        weights.update({
            f"h.{i}.attn.c_attn.weight": np.random.randn(n, 3 * n).astype(np.float32),
            f"h.{i}.attn.c_attn.bias": np.random.randn(3 * n).astype(np.float32),
            f"h.{i}.attn.c_proj.weight": np.random.randn(n, n).astype(np.float32),
            f"h.{i}.attn.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.mlp.c_fc.weight": np.random.randn(n, 4 * n).astype(np.float32),
            f"h.{i}.mlp.c_fc.bias": np.random.randn(4 * n).astype(np.float32),
            f"h.{i}.mlp.c_proj.weight": np.random.randn(4 * n, n).astype(np.float32),
            f"h.{i}.mlp.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.bias": np.random.randn(n).astype(np.float32),
        })
    return weights


def generate_full(model, token_ids, max_new_tokens):
    tokens = token_ids.copy()
    for _ in range(max_new_tokens):
        logits = model._forward_full(tokens)
        next_token = int(np.argmax(logits[0, -1, :]))
        tokens = np.concatenate([tokens, np.array([[next_token]], dtype=np.int32)], axis=1)
    return tokens


def benchmark():
    np.random.seed(42)

    print("=" * 90)
    print("GPT-2 Generation Benchmark: Full Recompute vs KV Cache")
    print("=" * 90)
    print("\nWarmup (Triton compile)...")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    weights = _make_random_weights(config)
    m = GPT2Model(config, weights)
    _ = m.generate(np.array([[5, 12]], dtype=np.int32), max_new_tokens=3, temperature=0.0)
    del m, config, weights
    print("Warmup done.\n")
    print("-" * 90)
    print(f"{'Prompt':<10} {'Gen':<8} {'Full (s)':<14} {'Cache (s)':<14} {'Speedup':<10} {'Quality':<10} {'Config':<20}")
    print("-" * 90)

    # Focused benchmark
    benchmarks = [
        # (config_name, config, prompt_len, gen_lens)
        ("Small", GPT2Config(n_layer=12, n_head=12, n_embd=768, vocab_size=50257, n_positions=1024),
         8, [1, 10, 30, 50]),
        ("Medium", GPT2Config(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, n_positions=1024),
         8, [1, 10, 30]),
        ("Large", GPT2Config(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, n_positions=1024),
         8, [1, 10]),
        ("XL", GPT2Config(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, n_positions=1024),
         8, [1, 10]),
    ]

    all_correct = True

    for cfg_name, config, plen, gen_lens in benchmarks:
        weights = _make_random_weights(config)
        prompt = np.random.randint(0, config.vocab_size, (1, plen)).astype(np.int32)

        for glen in gen_lens:
            # Full recompute
            m_full = GPT2Model(config, weights)
            t0 = time.time()
            out_full = generate_full(m_full, prompt.copy(), glen)
            t_full = time.time() - t0
            del m_full

            # KV cache
            m_cache = GPT2Model(config, weights)
            t0 = time.time()
            out_cache = m_cache.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
            t_cache = time.time() - t0
            del m_cache

            quality = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
            if quality == "FAIL":
                all_correct = False
            speedup = t_full / t_cache if t_cache > 0 else float('inf')
            print(f"{plen:<10} {glen:<8} {t_full:<14.4f} {t_cache:<14.4f} {speedup:<10.2f}x {quality:<10} {cfg_name:<20}")

    print("-" * 90)
    print(f"\nQuality: {'ALL PASS — cached output matches full recompute' if all_correct else 'SOME FAILED'}")


if __name__ == "__main__":
    benchmark()
