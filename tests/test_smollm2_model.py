"""Integration tests for SmolLM2 model forward pass and generation.

Uses a small random-weight model. Tests are runnable with::

    python tests/test_smollm2_model.py
"""

import numpy as np

from smollm2_triton.config import SmolLM2Config
from smollm2_triton.model import SmolLM2ForCausalLM


def _make_random_weights(config):
    """Create random numpy weights matching HF Llama weight shapes."""
    n = config.n_embd
    v = config.vocab_size
    n_kv = config.n_kv_head
    n_h = config.n_head
    d_k = n // n_h
    n_layer = config.n_layer
    ffn = config.n_ffn

    weights = {
        "model.embed_tokens.weight": np.random.randn(v, n).astype(np.float32) * 0.02,
        "model.norm.weight": np.random.randn(n).astype(np.float32) * 0.02,
    }
    for i in range(n_layer):
        weights.update({
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
    return weights


def test_model_forward():
    """Verify forward pass returns correct shape."""
    print("\n=== test_model_forward ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    token_ids = np.array([[5, 12, 7, 0, 3]], dtype=np.int32)  # (1, 5)
    logits = model.forward(token_ids)

    expected = (1, 5, 100)
    assert logits.shape == expected, f"Expected {expected}, got {logits.shape}"
    assert logits.dtype == np.float32
    print(f"  logits shape: {logits.shape}  [PASS]")
    return True


def test_model_generate():
    """Verify generate returns extended sequence with random weights."""
    print("\n=== test_model_generate ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    prompt = np.array([[5, 12, 7]], dtype=np.int32)  # (1, 3)
    out = model.generate(prompt, max_new_tokens=5, temperature=0.0)

    expected_len = 3 + 5  # 8
    assert out.shape == (1, expected_len), f"Expected (1, {expected_len}), got {out.shape}"
    assert out.dtype == np.int32
    print(f"  output shape: {out.shape}  [PASS]")
    return True


def test_model_kv_cache_equivalence():
    """Full forward (_forward_full) and cached prefill produce identical logits."""
    print("\n=== test_model_kv_cache_equivalence ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    token_ids = np.array([[5, 12, 7, 0, 3]], dtype=np.int32)  # (1, 5)

    # Full forward (no cache)
    logits_full = model.forward(token_ids, use_cache=False)

    # Cached forward (prefill)
    model._init_cache()
    logits_cached = model.forward(token_ids, use_cache=True)

    np.testing.assert_allclose(
        logits_full, logits_cached, rtol=1e-5, atol=1e-5,
        err_msg="Full forward and cached prefill logits must match",
    )
    print(f"  Logits match: max_diff={np.max(np.abs(logits_full - logits_cached)):.2e}  [PASS]")
    return True


def test_model_kv_cache_decode():
    """Verify that single-token decode logits match full forward logits."""
    print("\n=== test_model_kv_cache_decode ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    prompt = np.array([[5, 12, 7]], dtype=np.int32)

    # 1. Run prefill on prompt
    model._init_cache()
    _ = model.forward(prompt, use_cache=True)

    # 2. Run decode on next token
    next_token = np.array([[9]], dtype=np.int32)
    logits_decode = model.forward(next_token, use_cache=True)

    # 3. Run full forward on prompt + next token
    full_seq = np.concatenate([prompt, next_token], axis=1)
    logits_full = model.forward(full_seq, use_cache=False)

    # Compare the last token's logits
    np.testing.assert_allclose(
        logits_decode[:, -1, :], logits_full[:, -1, :],
        rtol=1e-5, atol=1e-5,
        err_msg="Decode logits must match full forward logits for the last token",
    )
    print("  Decode logits match full forward logits  [PASS]")
    return True


def test_model_deterministic():
    """Greedy (temperature=0) generation is deterministic with same seed."""
    print("\n=== test_model_deterministic ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    prompt = np.array([[1, 2, 3]], dtype=np.int32)

    out1 = model.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)
    out2 = model.generate(prompt.copy(), max_new_tokens=10, temperature=0.0)

    np.testing.assert_array_equal(out1, out2, err_msg="Greedy gen should be deterministic")
    print("  Greedy outputs match  [PASS]")
    return True


def test_model_with_temperature():
    """Temperature > 0 produces valid token IDs."""
    print("\n=== test_model_with_temperature ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    model = SmolLM2ForCausalLM(config, weights)
    prompt = np.array([[5, 12]], dtype=np.int32)
    out = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=50)

    assert out.shape == (1, 2 + 5)
    assert np.all(out >= 0) and np.all(out < config.vocab_size), \
        f"Tokens out of range: min={out.min()}, max={out.max()}"
    print(f"  Output shape: {out.shape}, tokens in [0, {config.vocab_size})  [PASS]")
    return True


def test_model_prealloc_cache_full_trip():
    """Pre-allocated KV cache: full prefill + multi-decode matches full recompute."""
    print("\n=== test_model_prealloc_cache_full_trip ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    # --- Cached: prefill + decode ---
    model_cache = SmolLM2ForCausalLM(config, weights)
    model_cache._init_cache()
    prompt = np.array([[5, 12, 7]], dtype=np.int32)
    _ = model_cache.forward(prompt, use_cache=True)

    tokens = prompt.copy()
    for _ in range(5):
        logits = model_cache.forward(
            np.array([[tokens[0, -1]]], dtype=np.int32), use_cache=True
        )
        next_token = int(np.argmax(logits[0, -1, :]))
        tokens = np.concatenate(
            [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
        )

    # --- Full recompute ---
    model_full = SmolLM2ForCausalLM(config, weights)
    full_tokens = prompt.copy()
    for _ in range(5):
        logits = model_full._forward_full(full_tokens)
        next_token = int(np.argmax(logits[0, -1, :]))
        full_tokens = np.concatenate(
            [full_tokens, np.array([[next_token]], dtype=np.int32)], axis=1
        )

    np.testing.assert_array_equal(
        tokens, full_tokens,
        err_msg="Pre-alloc KV cache generate and full recompute should produce identical tokens",
    )
    print(f"  Cached output:  {tokens.tolist()}")
    print(f"  Full recompute: {full_tokens.tolist()}  [PASS]")
    return True


def test_init_cache_rejects_invalid_max_seq():
    """_init_cache raises ValueError for non-positive max_seq."""
    print("\n=== test_init_cache_rejects_invalid_max_seq ===")
    config = SmolLM2Config(
        vocab_size=100, hidden_size=64,
        num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)
    model = SmolLM2ForCausalLM(config, weights)

    try:
        model._init_cache(max_seq=0)
        raise AssertionError("ValueError was not raised for max_seq=0")
    except ValueError as e:
        print(f"  max_seq=0 raised ValueError: {e}  [PASS]")

    try:
        model._init_cache(max_seq=-1)
        raise AssertionError("ValueError was not raised for max_seq=-1")
    except ValueError as e:
        print(f"  max_seq=-1 raised ValueError: {e}  [PASS]")

    # Valid case should still work
    model._init_cache(max_seq=128)
    assert model._cache_len == 0
    assert model.kv_cache[0]["k"].shape[1] == 128
    print(f"  max_seq=128 cache shape: {model.kv_cache[0]['k'].shape}  [PASS]")

    # Exceeding max_position_embeddings
    try:
        model._init_cache(max_seq=1024)
        raise AssertionError("ValueError was not raised for max_seq > max_position_embeddings")
    except ValueError as e:
        print(f"  max_seq > max_position_embeddings raised ValueError: {e}  [PASS]")

    return True


def test_model_rejects_empty_input():
    """Empty prompt raises ValueError."""
    print("\n=== test_model_rejects_empty_input ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)
    model = SmolLM2ForCausalLM(config, weights)
    token_ids = np.empty((1, 0), dtype=np.int32)

    try:
        model.generate(token_ids, max_new_tokens=1)
        raise AssertionError("ValueError was not raised for empty input")
    except ValueError as e:
        print(f"  ValueError raised as expected: {e}  [PASS]")
    return True


def test_gpu_resident_full_trip():
    """GPU-resident generate produces identical tokens to CPU generate (greedy)."""
    print("\n=== test_gpu_resident_full_trip ===")
    config = SmolLM2Config(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=512,
    )
    np.random.seed(42)
    weights = _make_random_weights(config)

    # GPU path — construction also goes through to_device(), catch GPU-unavailable
    prompt = np.array([[5, 12, 7]], dtype=np.int32)
    try:
        model_gpu = SmolLM2ForCausalLM(config, weights)
        out_gpu = model_gpu.generate_gpu(prompt, max_new_tokens=5, temperature=0.0)
    except RuntimeError as e:
        if "No supported GPU runtime found" in str(e):
            print("  SKIP: No GPU available  [SKIP]")
            return True  # Not a failure in CI without GPU
        raise

    # CPU path (deterministic with same seed)
    np.random.seed(42)
    model_cpu = SmolLM2ForCausalLM(config, weights)
    out_cpu = model_cpu.generate(prompt, max_new_tokens=5, temperature=0.0)

    np.testing.assert_array_equal(
        out_gpu, out_cpu,
        err_msg="GPU-resident generate must match CPU generate token-by-token",
    )
    print(f"  GPU output:  {out_gpu.tolist()}")
    print(f"  CPU output:  {out_cpu.tolist()}  [PASS]")
    return True


if __name__ == "__main__":
    print("Running SmolLM2 Model integration tests...")
    results = [
        ("test_model_forward", test_model_forward()),
        ("test_model_generate", test_model_generate()),
        ("test_model_kv_cache_equivalence", test_model_kv_cache_equivalence()),
        ("test_model_kv_cache_decode", test_model_kv_cache_decode()),
        ("test_model_deterministic", test_model_deterministic()),
        ("test_model_with_temperature", test_model_with_temperature()),
        ("test_model_prealloc_cache_full_trip", test_model_prealloc_cache_full_trip()),
        ("test_init_cache_rejects_invalid_max_seq", test_init_cache_rejects_invalid_max_seq()),
        ("test_model_rejects_empty_input", test_model_rejects_empty_input()),
        ("test_gpu_resident_full_trip", test_gpu_resident_full_trip()),
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
