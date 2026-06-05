"""
Unit tests for Triton GELU kernel (CUDA + HIP).

Tests numerical correctness against torch.nn.functional.gelu(approximate='tanh')
at various sizes including power-of-2, non-power-of-2, and single-element.
"""

import numpy as np
import torch
import torch.nn.functional as F
from gpt2_triton.kernels.gelu import gelu


def _gelu_ref_torch(x: np.ndarray) -> np.ndarray:
    """Reference GELU using PyTorch's tanh approximation."""
    x_t = torch.from_numpy(x)
    y_t = F.gelu(x_t, approximate="tanh")
    return y_t.numpy()


def test_gelu_correctness():
    """Test numerical correctness vs torch GELU at multiple sizes."""
    print("\n=== GELU Correctness Tests ===")

    test_sizes = [
        128,
        1024,
        777,   # non-power-of-2
        1,     # single element
    ]

    np.random.seed(42)
    all_passed = True
    for size in test_sizes:
        x = np.random.randn(size).astype(np.float32)
        out = gelu(x)
        ref = _gelu_ref_torch(x)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] N={size:>5} | max_diff={max_diff:.2e}")
        assert passed, f"GELU failed for N={size}, max_diff={max_diff:.2e}"
        all_passed &= passed

    return all_passed


def test_gelu_edge_cases():
    """Stability with extreme values and uniform arrays."""
    print("\n=== GELU Edge Case Tests ===")
    np.random.seed(0)

    # Large positive values (should saturate ~linear for x>>0).
    x_large = np.array([100.0, 1000.0, 1e6], dtype=np.float32)
    out_large = gelu(x_large)
    ref_large = _gelu_ref_torch(x_large)
    max_diff = float(np.abs(out_large - ref_large).max())
    passed = np.allclose(out_large, ref_large, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Large positive | max_diff={max_diff:.2e}")
    assert passed, f"Large positive GELU failed, max_diff={max_diff:.2e}"

    # Large negative values (should saturate to ~0).
    x_neg = np.array([-100.0, -1000.0, -1e6], dtype=np.float32)
    out_neg = gelu(x_neg)
    ref_neg = _gelu_ref_torch(x_neg)
    max_diff = float(np.abs(out_neg - ref_neg).max())
    passed = np.allclose(out_neg, ref_neg, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Large negative | max_diff={max_diff:.2e}")
    assert passed, f"Large negative GELU failed, max_diff={max_diff:.2e}"

    # Flat array (all same values).
    x_flat = np.full(256, 2.5, dtype=np.float32)
    out_flat = gelu(x_flat)
    ref_flat = _gelu_ref_torch(x_flat)
    max_diff = float(np.abs(out_flat - ref_flat).max())
    passed = np.allclose(out_flat, ref_flat, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Flat array  | max_diff={max_diff:.2e}")
    assert passed, f"Flat array GELU failed, max_diff={max_diff:.2e}"

    # All zeros.
    x_zero = np.zeros(128, dtype=np.float32)
    out_zero = gelu(x_zero)
    ref_zero = _gelu_ref_torch(x_zero)
    max_diff = float(np.abs(out_zero - ref_zero).max())
    passed = np.allclose(out_zero, ref_zero, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] All zeros   | max_diff={max_diff:.2e}")
    assert passed, f"All zeros GELU failed, max_diff={max_diff:.2e}"


def test_gelu_empty_array():
    """Empty input (N=0) should return an empty array."""
    print("\n=== GELU Empty Array Test ===")
    x = np.array([], dtype=np.float32)
    out = gelu(x)
    assert out.shape == (0,), f"Expected shape (0,), got {out.shape}"
    print("[PASS] Empty array handled correctly")


def test_gelu_multidim():
    """Multi-dimensional input should preserve shape."""
    print("\n=== GELU Multi-dimensional Test ===")
    np.random.seed(7)
    shape = (4, 16, 128)
    x = np.random.randn(*shape).astype(np.float32)
    out = gelu(x)
    ref = _gelu_ref_torch(x)
    assert out.shape == shape, f"Shape mismatch: {out.shape} vs {shape}"
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] shape={shape} | max_diff={max_diff:.2e}")
    assert passed, f"Multi-dim GELU failed, max_diff={max_diff:.2e}"


if __name__ == "__main__":
    print("Running GELU unit tests on current GPU backend...")
    test_gelu_correctness()
    test_gelu_edge_cases()
    test_gelu_empty_array()
    test_gelu_multidim()
    print("\n" + "=" * 45)
    print("All GELU tests PASSED")
    print("=" * 45)
