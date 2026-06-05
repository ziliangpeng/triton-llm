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
    """Empty input (N=0) should return an empty float32 array."""
    print("\n=== GELU Empty Array Test ===")
    # 1D empty array
    x = np.array([], dtype=np.float32)
    out = gelu(x)
    assert out.shape == (0,), f"Expected shape (0,), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    # Multi-dimensional empty array
    x_multi = np.empty((0, 10, 5), dtype=np.float32)
    out_multi = gelu(x_multi)
    assert out_multi.shape == (0, 10, 5), f"Expected shape (0, 10, 5), got {out_multi.shape}"
    assert out_multi.dtype == np.float32, f"Expected float32, got {out_multi.dtype}"

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


def test_gelu_non_contiguous():
    """Non-contiguous (strided) input should be handled via ascontiguousarray."""
    print("\n=== GELU Non-contiguous Input Test ===")
    # Create 2D matrix, take a column (strided slice).
    matrix = np.random.randn(8, 16).astype(np.float32)
    col = matrix[:, 3]  # non-contiguous column slice
    assert not col.flags["C_CONTIGUOUS"], "Test precondition failed: column slice should be non-contiguous"
    out = gelu(col)
    ref = _gelu_ref_torch(col)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Strided column  | max_diff={max_diff:.2e}")
    assert passed, f"Non-contiguous GELU failed, max_diff={max_diff:.2e}"

    # Also test a row slice (slice of first dimension).
    row = matrix[3, :]  # row is contiguous but let's also test for completeness
    out_row = gelu(row)
    ref_row = _gelu_ref_torch(row)
    passed_row = np.allclose(out_row, ref_row, atol=1e-4)
    print(f"[{'PASS' if passed_row else 'FAIL'}] Row slice       | max_diff={float(np.abs(out_row - ref_row).max()):.2e}")


def test_gelu_dtype_conversion():
    """Integer and float64 inputs should be converted to float32 without error."""
    print("\n=== GELU Dtype Conversion Test ===")

    # int32 input.
    x_int = np.array([-2, 0, 1, 3], dtype=np.int32)
    out_int = gelu(x_int)
    assert out_int.dtype == np.float32, f"Expected float32, got {out_int.dtype}"
    ref_int = _gelu_ref_torch(x_int.astype(np.float32))
    assert np.allclose(out_int, ref_int, atol=1e-4), "int32 GELU failed"

    # float64 input.
    x_f64 = np.array([-1.5, 0.0, 2.0], dtype=np.float64)
    out_f64 = gelu(x_f64)
    assert out_f64.dtype == np.float32, f"Expected float32, got {out_f64.dtype}"
    ref_f64 = _gelu_ref_torch(x_f64.astype(np.float32))
    assert np.allclose(out_f64, ref_f64, atol=1e-4), "float64 GELU failed"

    print("[PASS] int32 and float64 inputs correctly converted to float32")


if __name__ == "__main__":
    print("Running GELU unit tests on current GPU backend...")
    test_gelu_correctness()
    test_gelu_edge_cases()
    test_gelu_empty_array()
    test_gelu_multidim()
    test_gelu_non_contiguous()
    test_gelu_dtype_conversion()
    print("\n" + "=" * 45)
    print("All GELU tests PASSED")
    print("=" * 45)
