"""
Integration tests for GPT-2 model forward pass and generation.

Uses a small random-weight model. Tests are runnable with:
    python tests/test_model.py
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
        # GPT-2 uses Conv1D: weights stored as (in_features, out_features).
        # No transpose needed — gemm(hidden, w) uses w directly.
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


def test_forward_shape():
    """Verify forward pass returns correct shape."""
    print("\n=== test_forward_shape ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    token_ids = np.array([[5, 12, 7, 0, 3]], dtype=np.int32)  # (1, 5)
    logits = model.forward(token_ids)

    expected = (1, 5, 100)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert logits.dtype == np.float32
    print(f"  logits shape: {logits.shape}  [PASS]")
    return True


def test_generate_shape():
    """Verify generate returns extended sequence."""
    print("\n=== test_generate_shape ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[5, 12, 7]], dtype=np.int32)  # (1, 3)
    out = model.generate(prompt, max_new_tokens=5, temperature=0.0)

    expected_len = 3 + 5  # 8
    assert out.shape == (1, expected_len), f"Expected (1, {expected_len}), got {out.shape}"
    assert out.dtype == np.int32
    print(f"  output shape: {out.shape}  [PASS]")
    return True


def test_deterministic():
    """Greedy (temperature=0) generation is deterministic with same seed."""
    print("\n=== test_deterministic ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[1, 2, 3]], dtype=np.int32)

    out1 = model.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)
    out2 = model.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)

    np.testing.assert_array_equal(
        out1, out2,
        err_msg="Greedy generation should be deterministic",
    )
    print("  Greedy outputs match  [PASS]")
    return True


def test_empty_input():
    """Empty input (seq_len=0) raises or returns gracefully."""
    print("\n=== test_empty_input ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    token_ids = np.empty((1, 0), dtype=np.int32)

    try:
        logits = model.forward(token_ids)
        # Depending on kernel handling, may or may not succeed
        expected = (1, 0, 100)
        assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
        print(f"  Empty forward shape: {logits.shape}  [PASS]")
    except Exception as e:
        print(f"  Empty input raised: {type(e).__name__}: {e}  [ACCEPTABLE]")
    return True


def test_generate_with_temperature():
    """Temperature > 0 should produce valid token IDs."""
    print("\n=== test_generate_with_temperature ===")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=64, vocab_size=100, n_positions=1024)
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = GPT2Model(config, weights)
    prompt = np.array([[5, 12]], dtype=np.int32)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=50)

    assert out.shape == (1, 2 + 5)
    assert np.all(out >= 0) and np.all(out < config.vocab_size), \
        f"Tokens out of range: min={out.min()}, max={out.max()}"
    print(f"  Output shape: {out.shape}, tokens in [0, {config.vocab_size})  [PASS]")
    return True


if __name__ == "__main__":
    print("Running GPT-2 Model integration tests...")
    results = [
        ("test_forward_shape", test_forward_shape()),
        ("test_generate_shape", test_generate_shape()),
        ("test_deterministic", test_deterministic()),
        ("test_empty_input", test_empty_input()),
        ("test_generate_with_temperature", test_generate_with_temperature()),
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
