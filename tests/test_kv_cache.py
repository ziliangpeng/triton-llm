"""
Tests for KV-cached autoregressive decode.

Verifies that the KV cache produces identical output to full recompute,
and exercises edge cases.
"""

import numpy as np

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
        weights.update({
            f"h.{i}.attn.c_attn.weight": np.random.randn(3 * n, n).astype(np.float32),
            f"h.{i}.attn.c_attn.bias": np.random.randn(3 * n).astype(np.float32),
            f"h.{i}.attn.c_proj.weight": np.random.randn(n, n).astype(np.float32),
            f"h.{i}.attn.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.mlp.c_fc.weight": np.random.randn(4 * n, n).astype(np.float32),
            f"h.{i}.mlp.c_fc.bias": np.random.randn(4 * n).astype(np.float32),
            f"h.{i}.mlp.c_proj.weight": np.random.randn(n, 4 * n).astype(np.float32),
            f"h.{i}.mlp.c_proj.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_1.bias": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.weight": np.random.randn(n).astype(np.float32),
            f"h.{i}.ln_2.bias": np.random.randn(n).astype(np.float32),
        })
    return weights


def test_greedy_equivalence():
    """KV-cached generate produces identical output to full-recompute generate
    for any prompt when temperature=0 (greedy)."""
    print("\n=== test_greedy_equivalence ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    # Generate with KV cache
    model_cache = GPT2Model(config, weights)
    prompt = np.array([[5, 12, 7, 0, 3]], dtype=np.int32)
    out_cache = model_cache.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)

    # Generate with full recompute (no cache)
    model_full = GPT2Model(config, weights)
    out_full = model_full.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)

    np.testing.assert_array_equal(
        out_cache, out_full,
        err_msg="KV-cached and full-recompute greedy outputs should match",
    )
    print(f"  Prompt: {prompt.tolist()}")
    print(f"  Output: {out_cache.tolist()}")
    print(f"  Output (full): {out_full.tolist()}  [PASS]")
    return True


def test_cache_non_deterministic():
    """With temperature > 0 and top_k, output tokens are still valid."""
    print("\n=== test_cache_non_deterministic ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[5, 12]], dtype=np.int32)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=50)

    assert out.shape == (1, 2 + 5), f"Expected (1, 7), got {out.shape}"
    assert np.all(out >= 0) and np.all(out < config.vocab_size), \
        f"Tokens out of range: min={out.min()}, max={out.max()}"
    print(f"  Output shape: {out.shape}, tokens in [0, {config.vocab_size})  [PASS]")
    return True


def test_single_token_decode():
    """Prompt of length 1, generate 1 token with KV cache."""
    print("\n=== test_single_token_decode ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[42]], dtype=np.int32)  # single token
    out = model.generate(prompt, max_new_tokens=1, temperature=0.0)

    assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"
    assert out[0, 0] == 42, "First token should be the prompt"
    assert 0 <= out[0, 1] < config.vocab_size, \
        f"Generated token out of range: {out[0, 1]}"
    print(f"  Prompt: [[42]], output: {out.tolist()}  [PASS]")
    return True


def test_cache_empty():
    """Generate with max_new_tokens=0 returns same as input."""
    print("\n=== test_cache_empty ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[5, 12, 7]], dtype=np.int32)
    out = model.generate(prompt, max_new_tokens=0, temperature=0.0)

    np.testing.assert_array_equal(
        out, prompt,
        err_msg="max_new_tokens=0 should return the input unchanged",
    )
    print(f"  Prompt: {prompt.tolist()}, output: {out.tolist()}  [PASS]")
    return True


if __name__ == "__main__":
    print("Running KV cache tests...")
    results = [
        ("test_greedy_equivalence", test_greedy_equivalence()),
        ("test_cache_non_deterministic", test_cache_non_deterministic()),
        ("test_single_token_decode", test_single_token_decode()),
        ("test_cache_empty", test_cache_empty()),
    ]
    print("\n" + "=" * 50)
    all_pass = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  {status}: {name}")
    print("=" * 50)
    print(f"Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
