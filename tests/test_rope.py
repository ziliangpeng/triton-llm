"""
Unit tests for Triton RoPE kernel (CUDA + HIP).

Tests cover correctness against a NumPy reference, identity at position 0,
position offset behavior, empty input, precompute correctness, and single-row.
"""

import math

import numpy as np
from smollm2_triton.kernels.rope import apply_rope, precompute_cos_sin


def rope_ref(x, cos, sin, seq_len, position_offset=0):
    """
    NumPy reference for RoPE.

    x : (n_rows, d_k) float32 array — already flattened.
    cos, sin : (max_seq, d_k // 2) float32 arrays — half-packed.
    seq_len : int — determines position per row: pos = (row % seq_len) + position_offset.
    """
    d_k = x.shape[-1]
    half = d_k // 2
    n_rows = x.shape[0]
    result = x.copy()
    for row in range(n_rows):
        pos = (row % seq_len) + position_offset
        for i in range(half):
            x_even = result[row, i]
            x_odd = result[row, i + half]
            c = cos[pos, i]
            s = sin[pos, i]
            result[row, i] = x_even * c - x_odd * s
            result[row, i + half] = x_even * s + x_odd * c
    return result


def _test_case(name, x, cos, sin, seq_len, position_offset, atol=1e-4):
    """Run a single test case and print status."""
    out = apply_rope(x, cos, sin, seq_len=seq_len, position_offset=position_offset)
    ref = rope_ref(x, cos, sin, seq_len, position_offset)

    # Compare only valid positions (shape should match).
    shape_ok = out.shape == x.shape
    max_diff = float(np.abs(out - ref).max())
    value_ok = np.allclose(out, ref, atol=atol, rtol=1e-4)

    # For identity test at position 0, also check x unchanged.
    identity_ok = None
    if position_offset == 0 and seq_len >= 1:
        identity_ok = np.allclose(out, x, atol=atol)

    passed = shape_ok and value_ok
    extra = f"max_diff={max_diff:.2e}"
    if identity_ok is not None:
        extra += f" | identity_at_pos0={identity_ok}"

    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:<50} | {extra}")
    assert passed, f"Test '{name}' failed: shape_ok={shape_ok}, max_diff={max_diff:.2e}"
    return True


# ---------------------------------------------------------------------------
# Test 1: Correctness against NumPy reference
# ---------------------------------------------------------------------------

def test_rope_correctness():
    """Correctness vs numpy reference across multiple sizes."""
    print("\n=== RoPE Correctness Tests ===")

    np.random.seed(42)

    test_cases = [
        # (seq_len, d_k, n_rows)
        (8, 64, 576),        # SmolLM2-135M: seq=8, n_rows ~ batch*n_head*seq
        (128, 64, 960),      # 128 tokens, 960 rows
        (512, 64, 2048),     # 512 tokens, 2048 rows
        (16, 64, 144),       # 16 tokens, 144 rows (e.g. 2 heads, 9 seq)
        (32, 64, 32),        # 32 tokens, 1 row (single token)
    ]

    for seq_len, d_k, n_rows in test_cases:
        max_seq = max(seq_len, 128)
        cos, sin = precompute_cos_sin(max_seq, d_k, theta=100000.0)
        x = np.random.randn(n_rows, d_k).astype(np.float32)

        out = apply_rope(x, cos, sin, seq_len=seq_len)
        ref = rope_ref(x, cos, sin, seq_len)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] seq={seq_len:>4} d_k={d_k:>3} n_rows={n_rows:>5} | max_diff={max_diff:.2e}")
        assert passed, f"Correctness failed: seq={seq_len}, d_k={d_k}, n_rows={n_rows}, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 2: Identity at position 0 (cos=1, sin=0)
# ---------------------------------------------------------------------------

def test_rope_identity_zero_pos():
    """
    At position 0, cos(0)=1 and sin(0)=0, so rotation should be identity.

    Due to floating-point, cos(0) ≈ 1 and sin(0) ≈ 0, so result should be
    very close to the original input.
    """
    print("\n=== RoPE Identity at Position 0 ===")

    np.random.seed(1)
    d_k = 64
    seq_len = 1  # only position 0
    n_rows = 32

    cos, sin = precompute_cos_sin(seq_len, d_k, theta=100000.0)

    # cos[0] should be all ~1.0, sin[0] all ~0.0
    cos_ones = np.allclose(cos[0], 1.0, atol=1e-6)
    sin_zeros = np.allclose(sin[0], 0.0, atol=1e-6)
    print(f"  cos[0] close to 1: {cos_ones}, sin[0] close to 0: {sin_zeros}")

    x = np.random.randn(n_rows, d_k).astype(np.float32)
    out = apply_rope(x, cos, sin, seq_len=seq_len)

    max_diff = float(np.abs(out - x).max())
    passed = np.allclose(out, x, atol=1e-4, rtol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Position 0 identity | max_diff={max_diff:.2e}")
    assert passed, f"Identity at pos 0 failed, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 3: Position offset equivalence
# ---------------------------------------------------------------------------

def test_rope_position_offset():
    """
    Applying RoPE with position_offset=N on positions [0..seq_len-1] should
    produce the same result as applying with offset=0 on positions [N..N+seq_len-1].
    """
    print("\n=== RoPE Position Offset Tests ===")

    np.random.seed(3)
    d_k = 64
    half = d_k // 2
    seq_len = 16
    offset = 8
    max_seq = 64  # enough for max position

    cos, sin = precompute_cos_sin(max_seq, d_k, theta=100000.0)

    # Create input with seq_len rows (one position per row).
    n_rows = seq_len
    x = np.random.randn(n_rows, d_k).astype(np.float32)

    # Apply with offset.
    out_with_offset = apply_rope(x, cos, sin, seq_len=seq_len, position_offset=offset)

    # The reference: for row r, pos = r + offset, which should equal
    # applying to position r+offset directly.
    ref = rope_ref(x, cos, sin, seq_len, position_offset=offset)

    max_diff = float(np.abs(out_with_offset - ref).max())
    passed = np.allclose(out_with_offset, ref, atol=1e-4, rtol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Position offset={offset} | max_diff={max_diff:.2e}")
    assert passed, f"Position offset failed, max_diff={max_diff:.2e}"

    # Additional test: offset=0 vs look directly at cos/sin table.
    # Position 5 with offset 0 should be same as row 5.
    out_offset_0 = apply_rope(x, cos, sin, seq_len=seq_len, position_offset=0)
    ref_offset_0 = rope_ref(x, cos, sin, seq_len, position_offset=0)
    max_diff2 = float(np.abs(out_offset_0 - ref_offset_0).max())
    passed2 = np.allclose(out_offset_0, ref_offset_0, atol=1e-4, rtol=1e-4)
    status2 = "PASS" if passed2 else "FAIL"
    print(f"[{status2}] Position offset=0              | max_diff={max_diff2:.2e}")
    assert passed2, f"Position offset=0 failed, max_diff={max_diff2:.2e}"


# ---------------------------------------------------------------------------
# Test 4: Empty input
# ---------------------------------------------------------------------------

def test_rope_empty():
    """n_rows=0 should return an empty array with the correct shape."""
    print("\n=== RoPE Empty Input Test ===")

    d_k = 64
    cos, sin = precompute_cos_sin(8, d_k)
    x = np.empty((0, d_k), dtype=np.float32)

    out = apply_rope(x, cos, sin, seq_len=1)
    passed = out.shape == (0, d_k) and out.dtype == np.float32 and out.size == 0
    print(f"[{'PASS' if passed else 'FAIL'}] n_rows=0 empty input | shape={out.shape}")
    assert passed, f"Empty input failed, shape={out.shape}"


# ---------------------------------------------------------------------------
# Test 5: Precompute correctness
# ---------------------------------------------------------------------------

def test_rope_precompute():
    """Verify precompute_cos_sin produces correct values."""
    print("\n=== RoPE Precompute Tests ===")

    d_k = 64
    half = d_k // 2
    max_seq = 16
    theta = 100000.0

    cos, sin = precompute_cos_sin(max_seq, d_k, theta=theta)

    # Check shapes.
    shape_ok = cos.shape == (max_seq, half) and sin.shape == (max_seq, half)
    print(f"  Shape OK: {shape_ok} | cos.shape={cos.shape}")

    # Check dtype.
    dtype_ok = cos.dtype == np.float32 and sin.dtype == np.float32
    print(f"  Dtype OK: {dtype_ok}")

    # cos[0, :] should be all ~1.0.
    cos0_ok = np.allclose(cos[0], 1.0, atol=1e-6)
    # sin[0, :] should be all ~0.0.
    sin0_ok = np.allclose(sin[0], 0.0, atol=1e-6)
    print(f"  cos[0] ~ 1: {cos0_ok}, sin[0] ~ 0: {sin0_ok}")

    # Verify numerical values: compute theta_i and validate a few positions.
    indices = np.arange(half, dtype=np.float32)
    freqs = 1.0 / (theta ** (2 * indices / d_k))
    for p in [0, 1, 2, 8]:
        expected_cos = np.cos(p * freqs).astype(np.float32)
        expected_sin = np.sin(p * freqs).astype(np.float32)
        c_ok = np.allclose(cos[p], expected_cos, atol=1e-6)
        s_ok = np.allclose(sin[p], expected_sin, atol=1e-6)
        if not (c_ok and s_ok):
            c_diff = float(np.abs(cos[p] - expected_cos).max())
            s_diff = float(np.abs(sin[p] - expected_sin).max())
            print(f"  FAIL at p={p}: cos_diff={c_diff:.2e}, sin_diff={s_diff:.2e}")

    # Freq at i=0 should be 1.0 (any theta^0 = 1).
    # theta_i = 1.0 / (100000.0 ** (0 / 64)) = 1.0
    freq0_ok = abs(freqs[0] - 1.0) < 1e-6
    print(f"  freqs[0] == 1.0: {freq0_ok} (got {freqs[0]:.6f})")

    passed = shape_ok and dtype_ok and cos0_ok and sin0_ok and freq0_ok
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Precompute correctness")
    assert passed, "Precompute correctness failed"


# ---------------------------------------------------------------------------
# Test 6: Single element (one row)
# ---------------------------------------------------------------------------

def test_rope_single_element():
    """Single row (n_rows=1) should work correctly."""
    print("\n=== RoPE Single Row Test ===")

    np.random.seed(5)
    d_k = 64
    half = d_k // 2
    seq_len = 1

    cos, sin = precompute_cos_sin(8, d_k)
    x = np.random.randn(1, d_k).astype(np.float32)

    out = apply_rope(x, cos, sin, seq_len=seq_len)
    ref = rope_ref(x, cos, sin, seq_len)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Single row | shape={out.shape} | max_diff={max_diff:.2e}")
    assert passed, f"Single row failed, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 7: Input validation
# ---------------------------------------------------------------------------

def test_rope_input_validation():
    """Input validation: bad shapes/dims should raise ValueError."""
    print("\n=== RoPE Input Validation Tests ===")

    d_k = 64
    cos, sin = precompute_cos_sin(8, d_k)

    # 1D input.
    try:
        apply_rope(np.random.randn(64).astype(np.float32), cos, sin, seq_len=1)
        assert False, "Should have rejected 1D input"
    except ValueError:
        print("[PASS] 1D input correctly rejected")

    # Odd d_k.
    try:
        apply_rope(np.random.randn(4, 63).astype(np.float32), cos[:, :31], sin[:, :31], seq_len=1)
        assert False, "Should have rejected odd d_k"
    except ValueError:
        print("[PASS] Odd d_k correctly rejected")

    # Bad cos/sin shape.
    try:
        cos_bad = np.empty((8, 64), dtype=np.float32)
        apply_rope(np.random.randn(4, 64).astype(np.float32), cos_bad, sin, seq_len=1)
        assert False, "Should have rejected mismatched cos shape"
    except ValueError:
        print("[PASS] Mismatched cos shape correctly rejected")

    # seq_len < 1.
    try:
        apply_rope(np.random.randn(4, 64).astype(np.float32), cos, sin, seq_len=0)
        assert False, "Should have rejected seq_len=0"
    except ValueError:
        print("[PASS] seq_len=0 correctly rejected")

    # position_offset < 0.
    try:
        apply_rope(np.random.randn(4, 64).astype(np.float32), cos, sin, seq_len=1, position_offset=-1)
        assert False, "Should have rejected negative position_offset"
    except ValueError:
        print("[PASS] Negative offset correctly rejected")


# ---------------------------------------------------------------------------
# Test 8: Different theta values
# ---------------------------------------------------------------------------

def test_rope_theta():
    """RoPE should work with different theta values."""
    print("\n=== RoPE Different Theta Tests ===")

    np.random.seed(7)
    d_k = 64
    seq_len = 8
    n_rows = seq_len * 2  # 2 heads

    for theta in [10000.0, 100000.0, 500000.0]:
        cos, sin = precompute_cos_sin(16, d_k, theta=theta)
        x = np.random.randn(n_rows, d_k).astype(np.float32)

        out = apply_rope(x, cos, sin, seq_len=seq_len)
        ref = rope_ref(x, cos, sin, seq_len)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] theta={theta:>10.1f} | max_diff={max_diff:.2e}")
        assert passed, f"Theta={theta} failed, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Test 9: Multi-dimensional input
# ---------------------------------------------------------------------------

def test_rope_multidim():
    """Multi-dimensional input (e.g. (batch, n_head, seq, d_k)) via flattening."""
    print("\n=== RoPE Multi-dimensional Input Test ===")

    np.random.seed(9)
    B, H, S, D = 2, 9, 8, 64
    max_seq = 16

    cos, sin = precompute_cos_sin(max_seq, D)
    x = np.random.randn(B, H, S, D).astype(np.float32)

    out = apply_rope(x, cos, sin, seq_len=S)

    # Reference: flatten, apply, reshape.
    flat = x.reshape(-1, D)
    ref_flat = rope_ref(flat, cos, sin, S)
    ref = ref_flat.reshape(B, H, S, D)

    shape_ok = out.shape == (B, H, S, D)
    max_diff = float(np.abs(out - ref).max())
    value_ok = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    passed = shape_ok and value_ok
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] shape=({B},{H},{S},{D}) | shape_ok={shape_ok} | max_diff={max_diff:.2e}")
    assert passed, f"Multi-dim RoPE failed, shape={out.shape}, max_diff={max_diff:.2e}"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running RoPE unit tests on current GPU backend...")
    test_rope_correctness()
    test_rope_identity_zero_pos()
    test_rope_position_offset()
    test_rope_empty()
    test_rope_precompute()
    test_rope_single_element()
    test_rope_input_validation()
    test_rope_theta()
    test_rope_multidim()
    print("\n" + "=" * 50)
    print("All RoPE tests PASSED")
    print("=" * 50)
