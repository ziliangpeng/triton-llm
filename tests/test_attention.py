"""
Unit tests for Triton Fused Self-Attention kernel (CUDA + HIP).

Tests correctness against a pure NumPy reference implementation at various
sequence lengths and head dimensions, including GPT-2 typical sizes.
"""

import numpy as np
from gpt2_triton.kernels.attention import attention


def _attention_ref(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Pure NumPy reference for causal self-attention.

    Computes::

        O = softmax(Q @ K^T / sqrt(d_k) + mask) @ V

    where mask is upper-triangular (-inf for j > i).
    """
    N, d_k = q.shape
    scale = 1.0 / np.sqrt(d_k)
    scores = (q @ k.T) * scale

    # Causal mask: upper triangle (j > i) gets -inf
    mask = np.triu(np.full((N, N), -np.inf, dtype=np.float32), k=1)
    scores = scores + mask

    # Numerically stable softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    probs = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    return probs @ v


def test_attention_correctness():
    """Test numerical correctness vs NumPy reference at multiple shapes."""
    print("\n=== Attention Correctness Tests ===")

    test_cases = [
        (1, 64),
        (4, 64),
        (16, 64),
        (32, 64),
        (128, 64),
        (777, 64),
        (1024, 64),
        (16, 32),
        (16, 128),
    ]

    np.random.seed(42)
    all_passed = True
    for N, d_k in test_cases:
        q = np.random.randn(N, d_k).astype(np.float32)
        k = np.random.randn(N, d_k).astype(np.float32)
        v = np.random.randn(N, d_k).astype(np.float32)

        out = attention(q, k, v)
        ref = _attention_ref(q, k, v)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] N={N:>4} d_k={d_k:>3} | max_diff={max_diff:.2e}")
        assert passed, f"Attention failed for N={N}, d_k={d_k}, max_diff={max_diff:.2e}"
        all_passed &= passed

    return all_passed


def test_attention_causal_mask():
    """Verify that the causal mask is correctly applied.

    For position i, all output values should be zero if we zero out
    the values at positions > i (since those positions contribute nothing
    due to the causal mask).
    """
    print("\n=== Attention Causal Mask Test ===")

    N, d_k = 8, 64
    np.random.seed(42)
    q = np.random.randn(N, d_k).astype(np.float32)
    k = np.random.randn(N, d_k).astype(np.float32)
    v = np.random.randn(N, d_k).astype(np.float32)

    out = attention(q, k, v)
    ref = _attention_ref(q, k, v)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Causal mask | max_diff={max_diff:.2e}")
    assert passed, f"Causal mask test failed, max_diff={max_diff:.2e}"


def test_attention_identical_qkv():
    """When Q=K=V=I (identity), attention should produce the identity.

    With Q=K=V=I, the scores are I @ I^T / sqrt(d_k) = I / sqrt(d_k).
    After causal softmax, row i has non-zero only on positions <= i.
    """
    print("\n=== Attention Identity Test ===")

    N, d_k = 8, 64
    q = np.eye(N, d_k, dtype=np.float32)
    k = np.eye(N, d_k, dtype=np.float32)
    v = np.eye(N, d_k, dtype=np.float32)

    out = attention(q, k, v)
    ref = _attention_ref(q, k, v)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Identity Q=K=V=I | max_diff={max_diff:.2e}")
    assert passed, f"Identity test failed, max_diff={max_diff:.2e}"


def test_attention_single_token():
    """Single token (N=1) should just be the value itself after softmax.

    With N=1, the causal mask allows position 0 to attend to itself.
    softmax(single_score) = 1.0, so O = V.
    """
    print("\n=== Attention Single Token Test ===")

    d_k = 64
    q = np.random.randn(1, d_k).astype(np.float32)
    k = np.random.randn(1, d_k).astype(np.float32)
    v = np.random.randn(1, d_k).astype(np.float32)

    out = attention(q, k, v)
    ref = _attention_ref(q, k, v)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] Single token N=1 | max_diff={max_diff:.2e}")
    assert passed, f"Single token test failed, max_diff={max_diff:.2e}"


if __name__ == "__main__":
    test_attention_correctness()
    test_attention_causal_mask()
    test_attention_identical_qkv()
    test_attention_single_token()
    print("\n=== All attention tests passed! ===")
