#!/usr/bin/env python3
"""Quick benchmark: compare pre-alloc KV cache with known old-cache baseline."""
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

    print(f"Model: SmolLM2-135M (30L, 9H, 3KV, 576)")
    print()

    # Warmup: 2 tokens to compile Triton kernels
    print("Warmup (2 tokens)...")
    _ = model.generate(prompt.copy(), max_new_tokens=2, temperature=0.0)
    print("  done")

    # Benchmark: 50 tokens, 1 run (Triton kernels now hot-compiled)
    print(f"\nBenchmark: 8 prompt + 50 gen (58 total)")
    t0 = time.perf_counter()
    out = model.generate(prompt.copy(), max_new_tokens=50, temperature=0.0)
    t = time.perf_counter() - t0
    print(f"  Total: {t:.3f}s")
    print(f"  Steps: 1 prefill + 50 decode = 51 steps")
    print(f"  Per step: {t/51*1000:.1f}ms")
    assert out.shape[1] == 58, f"Expected 58, got {out.shape[1]}"
    print(f"  Output len: {out.shape[1]} [OK]")

    # Longer: 200 tokens
    print(f"\nBenchmark: 8 prompt + 200 gen (208 total)")
    t0 = time.perf_counter()
    out = model.generate(prompt.copy(), max_new_tokens=200, temperature=0.0)
    t = time.perf_counter() - t0
    print(f"  Total: {t:.3f}s")
    print(f"  Steps: 1 + 200 = 201")
    print(f"  Per step: {t/201*1000:.1f}ms")
    assert out.shape[1] == 208, f"Expected 208, got {out.shape[1]}"
    print(f"  Output len: {out.shape[1]} [OK]")

    print("\n✅ Benchmark complete")


if __name__ == "__main__":
    main()
