#!/usr/bin/env python3
"""Benchmark: full-recompute vs KV-cached generation on GPT-2.

Usage:
    srun --gres=gpu:1 -N 1 python scripts/benchmark_generate.py

Tests all 4 GPT-2 config sizes across multiple generation lengths.
Reports latency, speedup, and quality equivalence.
"""

import time
import numpy as np
import sys
sys.path.insert(0, '.')

from gpt2_triton.config import GPT2Config
from gpt2_triton.model import GPT2Model


def _make_random_weights(config):
    """Create random numpy weights matching HF GPT-2 weight shapes."""
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
        # GPT-2 uses Conv1D: (in, out) — no transpose
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


def generate_full_recompute(model, token_ids, max_new_tokens, temperature=0.0, top_k=0):
    """Full-recompute generation (old way, no cache)."""
    tokens = token_ids.copy()
    for _ in range(max_new_tokens):
        logits = model._forward_full(tokens)
        next_logits = logits[0, -1, :]
        if temperature > 1e-6:
            scaled = next_logits / temperature
            from gpt2_triton.kernels.softmax import softmax
            probs = softmax(scaled.reshape(1, -1)).ravel()
            if top_k > 0:
                indices = np.argpartition(probs, -top_k)[-top_k:]
                filtered = np.zeros_like(probs)
                filtered[indices] = probs[indices]
                filtered /= filtered.sum()
                probs = filtered
            next_token = int(np.random.choice(len(probs), p=probs))
        else:
            next_token = int(np.argmax(next_logits))
        tokens = np.concatenate(
            [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
        )
    return tokens


def benchmark():
    np.random.seed(42)

    configs = [
        ("GPT-2 Small",  GPT2Config(n_layer=12, n_head=12, n_embd=768,  vocab_size=50257, n_positions=1024)),
        ("GPT-2 Medium", GPT2Config(n_layer=24, n_head=16, n_embd=1024, vocab_size=50257, n_positions=1024)),
        ("GPT-2 Large",  GPT2Config(n_layer=36, n_head=20, n_embd=1280, vocab_size=50257, n_positions=1024)),
        ("GPT-2 XL",     GPT2Config(n_layer=48, n_head=25, n_embd=1600, vocab_size=50257, n_positions=1024)),
    ]

    prompt_lens = [8, 32, 128]
    gen_lens = [1, 10, 30, 50]

    print("=" * 100)
    print("GPT-2 Generation: Full Recompute vs KV Cache")
    print("=" * 100)
    print(f"\n{'Config':<20} {'Prompt':<8} {'Gen':<6} {'Full (s)':<12} {'Cache (s)':<12} {'Speedup':<10} {'Quality':<10}")
    print("-" * 100)

    all_correct = True

    for name, config in configs:
        weights = _make_random_weights(config)
        # Warmup / Triton compile
        _ = GPT2Model(config, weights)
        del _

        for plen in prompt_lens:
            for glen in gen_lens:
                prompt = np.random.randint(0, config.vocab_size, (1, plen)).astype(np.int32)

                # --- Full recompute ---
                model_full = GPT2Model(config, weights)
                t0 = time.time()
                out_full = generate_full_recompute(model_full, prompt.copy(), glen, temperature=0.0)
                t_full = time.time() - t0
                del model_full

                # --- KV cache ---
                model_cache = GPT2Model(config, weights)
                t0 = time.time()
                out_cache = model_cache.generate(prompt.copy(), max_new_tokens=glen, temperature=0.0)
                t_cache = time.time() - t0
                del model_cache

                # Quality: greedy outputs must match
                quality = "PASS" if np.array_equal(out_full, out_cache) else "FAIL"
                if quality == "FAIL":
                    all_correct = False

                speedup = t_full / t_cache if t_cache > 0 else float('inf')
                print(f"{name:<20} {plen:<8} {glen:<6} {t_full:<12.4f} {t_cache:<12.4f} {speedup:<10.2f}x {quality:<10}")

    print("-" * 100)
    print(f"\nQuality check: {'ALL PASS' if all_correct else 'SOME FAILED'} — greedy output matches across all configs")

    print("\nNote: Full recompute time grows with prompt length (each step processes the")
    print("entire growing sequence). KV cache decode time is roughly constant per step.")


if __name__ == "__main__":
    benchmark()
