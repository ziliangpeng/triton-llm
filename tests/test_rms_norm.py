"""
Unit tests for Triton RMSNorm kernel (CUDA + HIP).
"""

import numpy as np
from smollm2_triton.kernels.rms_norm import rms_norm


def rms_norm_ref(x, weight, eps=1e-5):
    """NumPy reference RMSNorm over the last dim."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def test_rms_norm_correctness():
    """Correctness vs numpy reference across several shapes (incl. non-power-of-2 N)."""
    print("\n=== RMSNorm Correctness Tests ===")

    test_cases = [
        (8, 64),
        (16, 128),
        (32, 576),      # SmolLM2-135M hidden size (non-power-of-2)
        (16, 960),      # SmolLM2-360M hidden size (non-power-of-2)
        (8, 2048),      # SmolLM2-1.7B hidden size (power-of-2)
        (16, 777),      # Random non-power-of-2
        (4, 1),         # Single-element last dimension
    ]

    np.random.seed(42)
    for M, N in test_cases:
        x = np.random.randn(M, N).astype(np.float32)
        weight = np.random.randn(N).astype(np.float32)

        out = rms_norm(x, weight, eps=1e-5)
        ref = rms_norm_ref(x, weight, eps=1e-5)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] M={M:>4} N={N:>5} | max_diff={max_diff:.2e}")
        assert passed, f"RMSNorm failed for M={M}, N={N}, max_diff={max_diff:.2e}"


def test_rms_norm_edge_cases():
    """Test edge cases: large values, all zeros, single element."""
    print("\n=== RMSNorm Edge Case Tests ===")

    # Large values (but within float32 range)
    M, N = 8, 576
    np.random.seed(7)
    x = np.random.randn(M, N).astype(np.float32) * 1000.0
    weight = np.ones(N, dtype=np.float32)

    out = rms_norm(x, weight, eps=1e-5)
    ref = rms_norm_ref(x, weight, eps=1e-5)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Large values  | max_diff={max_diff:.2e}")
    assert passed, f"Large-value RMSNorm failed, max_diff={max_diff:.2e}"

    # All zeros: RMS = sqrt(eps), output should be (0 / sqrt(eps)) * weight = 0
    x = np.zeros((M, N), dtype=np.float32)
    weight = np.random.randn(N).astype(np.float32)
    out = rms_norm(x, weight, eps=1e-5)
    ref = rms_norm_ref(x, weight, eps=1e-5)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    # Both should be all-zeros (weight * 0 = 0)
    zero_passed = np.all(out == 0.0)
    print(f"[{'PASS' if zero_passed else 'FAIL'}] All zeros    | max_diff={max_diff:.2e} | all_zero={np.all(out == 0.0)}")
    assert passed, f"All-zero RMSNorm failed, max_diff={max_diff:.2e}"

    # Single element (M=1, N=1)
    x = np.array([[3.0]], dtype=np.float32)
    weight = np.array([2.0], dtype=np.float32)
    out = rms_norm(x, weight, eps=1e-5)
    # Manual: rms = sqrt(9 + 1e-5) ≈ 3.0, y = (3/3)*2 = 2.0
    ref = rms_norm_ref(x, weight, eps=1e-5)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Single element | max_diff={max_diff:.2e} | val={out.item():.4f}")
    assert passed, f"Single-element RMSNorm failed, max_diff={max_diff:.2e}"


def test_rms_norm_empty():
    """M=0 should return an empty array with the correct shape."""
    print("\n=== RMSNorm Empty Input Test ===")
    x = np.empty((0, 576), dtype=np.float32)
    weight = np.ones(576, dtype=np.float32)

    out = rms_norm(x, weight, eps=1e-5)
    passed = out.shape == (0, 576) and out.dtype == np.float32 and out.size == 0
    print(f"[{'PASS' if passed else 'FAIL'}] M=0 empty input | shape={out.shape}")
    assert passed, f"Empty input failed, shape={out.shape}"


def test_rms_norm_multidim():
    """Multi-dimensional input (B, S, N) preserves shape via leading-dim flattening."""
    print("\n=== RMSNorm Multi-dimensional Input Test ===")
    B, S, N = 4, 16, 576
    np.random.seed(8)
    x = np.random.randn(B, S, N).astype(np.float32)
    weight = np.random.randn(N).astype(np.float32)

    out = rms_norm(x, weight, eps=1e-5)
    ref = rms_norm_ref(x, weight, eps=1e-5)

    shape_passed = out.shape == (B, S, N)
    max_diff = float(np.abs(out - ref).max())
    value_passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    passed = shape_passed and value_passed
    print(f"[{'PASS' if passed else 'FAIL'}] shape=({B},{S},{N}) | shape_ok={shape_passed} | max_diff={max_diff:.2e}")
    assert passed, f"Multi-dim RMSNorm failed, shape={out.shape}, max_diff={max_diff:.2e}"


def test_rms_norm_weight_shape():
    """Mismatched weight shape should raise ValueError."""
    print("\n=== RMSNorm Weight Shape Validation ===")
    x = np.random.randn(8, 576).astype(np.float32)

    # Wrong weight shape.
    try:
        rms_norm(x, np.ones(64, dtype=np.float32))
        assert False, "Should have rejected mismatched weight shape"
    except ValueError:
        print("[PASS] Mismatched weight shape correctly rejected")

    # 1D input should be rejected.
    try:
        rms_norm(np.random.randn(64).astype(np.float32), np.ones(64, dtype=np.float32))
        assert False, "Should have rejected 1D input"
    except ValueError:
        print("[PASS] 1D input correctly rejected")

    # N=0 last dimension.
    try:
        rms_norm(np.random.randn(4, 0).astype(np.float32), np.ones(0, dtype=np.float32))
        assert False, "Should have rejected N=0"
    except ValueError:
        print("[PASS] N=0 correctly rejected")


if __name__ == "__main__":
    print("Running RMSNorm unit tests on current GPU backend...")
    test_rms_norm_correctness()
    test_rms_norm_edge_cases()
    test_rms_norm_empty()
    test_rms_norm_multidim()
    test_rms_norm_weight_shape()
    print("\n" + "=" * 45)
    print("All RMSNorm tests PASSED")
    print("=" * 45)
