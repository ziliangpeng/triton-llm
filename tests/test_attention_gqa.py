"""
Unit tests for Triton GQA (Grouped Query Attention) kernel.

Tests correctness against a NumPy reference implementation at various
configurations including SmolLM2 model sizes.
"""

import numpy as np
from smollm2_triton.kernels.attention_gqa import attention_gqa
from smollm2_triton.kernels.rope import precompute_cos_sin, apply_rope


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _causal_mask(seq_q: int, seq_k: int) -> np.ndarray:
    """Create causal mask: -inf for positions j > i, 0 elsewhere.

    Each query position i attends only to key positions j <= i.
    For decode mode (seq_q=1, seq_k>1), the single query attends to all past KV.
    """
    mask = np.zeros((seq_q, seq_k), dtype=np.float32)
    for i in range(seq_q):
        j_limit = min(i + 1, seq_k)
        if j_limit < seq_k:
            mask[i, j_limit:] = -np.inf
    return mask


def attention_gqa_ref(
    q_flat: np.ndarray,
    k_flat: np.ndarray,
    v_flat: np.ndarray,
    n_head: int,
    n_kv_head: int,
    causal: bool = True,
) -> np.ndarray:
    """NumPy reference for GQA by expanding K/V to full n_head.

    Parameters
    ----------
    q_flat : (n_head * seq_q, d_k) float32 — flat input.
    k_flat : (n_kv_head * seq_k, d_k) float32.
    v_flat : (n_kv_head * seq_k, d_k) float32.
    n_head : int
    n_kv_head : int
    causal : bool

    Returns
    -------
    o : (seq_q, n_head, d_k) float32 — per-head output.
    """
    group_size = n_head // n_kv_head
    d_k = q_flat.shape[-1]
    seq_q = q_flat.shape[0] // n_head
    seq_k = k_flat.shape[0] // n_kv_head

    # Reshape to (n_head, seq_q, d_k) and (n_kv_head, seq_k, d_k)
    q_3d = q_flat.reshape(n_head, seq_q, d_k)
    k_3d = k_flat.reshape(n_kv_head, seq_k, d_k)
    v_3d = v_flat.reshape(n_kv_head, seq_k, d_k)

    # Expand K/V: repeat each kv head group_size times along head axis
    k_expanded = np.repeat(k_3d, group_size, axis=0)  # (n_head, seq_k, d_k)
    v_expanded = np.repeat(v_3d, group_size, axis=0)   # (n_head, seq_k, d_k)

    scale = 1.0 / np.sqrt(d_k)

    outputs = []
    for h in range(n_head):
        # Q: (seq_q, d_k), K: (seq_k, d_k) → scores: (seq_q, seq_k)
        scores = q_3d[h] @ k_expanded[h].T * scale

        if causal:
            scores = scores + _causal_mask(seq_q, seq_k)

        attn = _softmax(scores, axis=-1)
        o_h = attn @ v_expanded[h]  # (seq_q, d_k)
        outputs.append(o_h)

    return np.stack(outputs, axis=0).reshape(n_head * seq_q, d_k)  # (n_head * seq_q, d_k) flat


# =============================================================================
# Tests
# =============================================================================


def test_gqa_equivalence():
    """GQA with n_head == n_kv_head produces same result as regular MHA."""
    print("\n=== Test: GQA Equivalence (n_head == n_kv_head) ===")

    np.random.seed(42)
    n_head = 4
    n_kv_head = 4
    d_k = 64
    seq = 16

    q = np.random.randn(n_head * seq, d_k).astype(np.float32)
    k = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)
    v = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)

    out = attention_gqa(q, k, v, n_head, n_kv_head, causal=True)
    ref = attention_gqa_ref(q, k, v, n_head, n_kv_head, causal=True)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] n_head={n_head}, n_kv_head={n_kv_head} | max_diff={max_diff:.2e}")
    assert passed, f"GQA equivalence test failed, max_diff={max_diff:.2e}"


def test_gqa_reduced_kv():
    """GQA with n_kv_head < n_head (SmolLM2-like: 9→3, 15→5, 32→8)."""
    print("\n=== Test: GQA Reduced KV ===")

    np.random.seed(42)
    configs = [
        (9, 3, 64, 8),    # SmolLM2-135M-like
        (15, 5, 64, 8),   # SmolLM2-360M-like
        (32, 8, 64, 16),  # Generic
        (6, 2, 128, 8),   # Different d_k
    ]

    all_passed = True
    for n_head, n_kv_head, d_k, seq in configs:
        q = np.random.randn(n_head * seq, d_k).astype(np.float32)
        k = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)
        v = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)

        out = attention_gqa(q, k, v, n_head, n_kv_head, causal=True)
        ref = attention_gqa_ref(q, k, v, n_head, n_kv_head, causal=True)
        ref = ref.reshape(n_head * seq, d_k)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] n_head={n_head}, n_kv_head={n_kv_head}, d_k={d_k}, seq={seq} | max_diff={max_diff:.2e}")
        assert passed, f"Reduced KV test failed: n_head={n_head}, n_kv_head={n_kv_head}, d_k={d_k}, seq={seq}, max_diff={max_diff:.2e}"
        all_passed &= passed

    return all_passed


def test_gqa_with_rope():
    """End-to-end: apply_rope + gqa_attention."""
    print("\n=== Test: GQA with RoPE ===")

    np.random.seed(42)
    n_head = 9
    n_kv_head = 3
    d_k = 64
    seq = 8
    max_seq = 64
    rope_theta = 100000.0

    # Precompute cos/sin
    cos, sin = precompute_cos_sin(max_seq, d_k, theta=rope_theta)

    # Create Q, K in flat format
    q = np.random.randn(n_head * seq, d_k).astype(np.float32)
    k = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)
    v = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)

    # Apply RoPE
    q_rope = apply_rope(q, cos, sin, seq_len=seq)
    k_rope = apply_rope(k, cos, sin, seq_len=seq)

    # GQA attention
    out = attention_gqa(q_rope, k_rope, v, n_head, n_kv_head, causal=True)

    # Reference: same RoPE + reference attention
    ref = attention_gqa_ref(q_rope, k_rope, v, n_head, n_kv_head, causal=True)
    ref = ref.reshape(n_head * seq, d_k)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] n_head={n_head}, n_kv_head={n_kv_head} | max_diff={max_diff:.2e}")
    assert passed, f"GQA with RoPE test failed, max_diff={max_diff:.2e}"


def test_gqa_causal():
    """Causal masking: modifying future K/V positions does NOT affect earlier outputs."""
    print("\n=== Test: GQA Causal Mask ===")

    np.random.seed(42)
    n_head = 6
    n_kv_head = 2
    d_k = 64
    seq = 8

    q = np.random.randn(n_head * seq, d_k).astype(np.float32)
    k = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)
    v = np.random.randn(n_kv_head * seq, d_k).astype(np.float32)

    # Reference output
    ref = attention_gqa(q, k, v, n_head, n_kv_head, causal=True)

    # Perturb the last key position for all KV heads (correct GQA flat layout: 
    # rows are (n_kv_head, seq, d_k), so perturb row (h*seq+seq-1, :) for each h)
    k_pert = k.copy()
    for h in range(n_kv_head):
        k_pert[h * seq + (seq - 1), :] = np.random.randn(d_k).astype(np.float32)
    out = attention_gqa(q, k_pert, v, n_head, n_kv_head, causal=True)

    # Reshape to (n_head, seq, d_k) to compare per-position
    ref_r = ref.reshape(n_head, seq, d_k)
    out_r = out.reshape(n_head, seq, d_k)

    # All positions except the last should be unchanged
    unchanged = np.allclose(out_r[:, :-1, :], ref_r[:, :-1, :], atol=1e-5)
    status = "PASS" if unchanged else "FAIL"
    print(f"  [{status}] Positions 0..{seq-2} unchanged: {unchanged}")
    assert unchanged, "Causal mask broken — earlier positions changed"

    print(f"  [INFO] Last position may differ (expected)")


def test_gqa_non_causal():
    """Non-causal / decode mode (seq_q=1, seq_k=large) — single query attends to all KV."""
    print("\n=== Test: GQA Non-Causal / Decode Mode ===")

    np.random.seed(42)
    n_head = 9
    n_kv_head = 3
    d_k = 64
    seq_q = 1
    seq_k = 16  # KV cache size

    q = np.random.randn(n_head * seq_q, d_k).astype(np.float32)
    k = np.random.randn(n_kv_head * seq_k, d_k).astype(np.float32)
    v = np.random.randn(n_kv_head * seq_k, d_k).astype(np.float32)

    out = attention_gqa(q, k, v, n_head, n_kv_head, causal=False)
    ref = attention_gqa_ref(q, k, v, n_head, n_kv_head, causal=False)
    ref = ref.reshape(n_head * seq_q, d_k)

    assert out.shape == (n_head * seq_q, d_k), \
        f"Expected ({n_head * seq_q}, {d_k}), got {out.shape}"
    assert not np.any(np.isnan(out)), "Output contains NaN"

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] seq_q={seq_q}, seq_k={seq_k} | max_diff={max_diff:.2e}")
    assert passed, f"Non-causal GQA failed, max_diff={max_diff:.2e}"


def test_gqa_empty():
    """Zero-length inputs should return empty output."""
    print("\n=== Test: GQA Empty Input ===")

    n_head = 4
    n_kv_head = 2
    d_k = 64

    # seq_q=0
    q_empty = np.empty((0, d_k), dtype=np.float32)
    k_some = np.random.randn(n_kv_head * 8, d_k).astype(np.float32)
    v_some = np.random.randn(n_kv_head * 8, d_k).astype(np.float32)
    out = attention_gqa(q_empty, k_some, v_some, n_head, n_kv_head, causal=False)
    expected_shape = (n_head * 0, d_k)
    shape_ok = out.shape == expected_shape
    status = "PASS" if shape_ok else "FAIL"
    print(f"  [{status}] seq_q=0: shape {out.shape} == {expected_shape}: {shape_ok}")
    assert shape_ok, f"seq_q=0: expected {expected_shape}, got {out.shape}"

    # seq_k=0
    q_some = np.random.randn(n_head * 4, d_k).astype(np.float32)
    k_empty = np.empty((0, d_k), dtype=np.float32)
    v_empty = np.empty((0, d_k), dtype=np.float32)
    out2 = attention_gqa(q_some, k_empty, v_empty, n_head, n_kv_head, causal=False)
    expected_shape2 = (n_head * 4, d_k)
    shape_ok2 = out2.shape == expected_shape2 and np.all(out2 == 0.0)
    status2 = "PASS" if shape_ok2 else "FAIL"
    print(f"  [{status2}] seq_k=0: shape {out2.shape} == {expected_shape2}: {shape_ok2}")
    assert shape_ok2, f"seq_k=0: expected {expected_shape2} and all zeros, got shape {out2.shape}"

    # Both zero
    out3 = attention_gqa(q_empty, k_empty, v_empty, n_head, n_kv_head, causal=False)
    expected_shape3 = (0, d_k)
    shape_ok3 = out3.shape == expected_shape3
    status3 = "PASS" if shape_ok3 else "FAIL"
    print(f"  [{status3}] both zero: shape {out3.shape} == {expected_shape3}: {shape_ok3}")
    assert shape_ok3, f"both zero: expected {expected_shape3}, got {out3.shape}"


if __name__ == "__main__":
    test_gqa_equivalence()
    test_gqa_reduced_kv()
    test_gqa_with_rope()
    test_gqa_causal()
    test_gqa_non_causal()
    test_gqa_empty()
    print("\n=== All GQA attention tests passed! ===")
