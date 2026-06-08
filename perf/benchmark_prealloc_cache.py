#!/usr/bin/env python3
"""Benchmark: pre-allocated KV cache vs old concat-based cache for SmolLM2.

Usage:
    srun --gres=gpu:1 bash -c 'PYTHONPATH=. python perf/benchmark_prealloc_cache.py'
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


def main():
    config = SmolLM2Config(
        hidden_size=576, num_hidden_layers=30,
        num_attention_heads=9, num_key_value_heads=3,
        intermediate_size=1536, vocab_size=49152,
        max_position_embeddings=2048,
    )
    np.random.seed(42)
    weights = make_weights(config)
    prompt = np.array([[5] * 8], dtype=np.int32)

    model = SmolLM2ForCausalLM(config, weights)

    print("=== Pre-allocated KV Cache Performance ===")
    print(f"Model: SmolLM2-135M (30 layers, 9 heads, 3 kv heads, 576 embd)")
    print()

    # Warmup
    print("Warmup...")
    _ = model.generate(prompt.copy(), max_new_tokens=5, temperature=0.0)

    for gen_tokens in [10, 50, 100, 200]:
        times = []
        for run in range(3):
            t0 = time.perf_counter()
            out = model.generate(prompt.copy(), max_new_tokens=gen_tokens, temperature=0.0)
            t = time.perf_counter() - t0
            times.append(t)
            print(f"  gen={gen_tokens}, run {run+1}: {t:.3f}s, output len={out.shape[1]}")

        avg = sum(times) / len(times)
        total_seq = 8 + gen_tokens
        per_token = avg / (1 + gen_tokens) * 1000  # prefill + gen_tokens decode steps
        print(f"  => avg={avg:.3f}s, per_step={per_token:.1f}ms")
        print()


if __name__ == "__main__":
    main()
