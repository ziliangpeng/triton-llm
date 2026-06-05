"""
Unit tests for Triton element-wise Add kernel (CUDA + HIP).

Tests numerical correctness against NumPy reference (x + y) at various
sizes, edge cases, multi-dimensional inputs, non-contiguous buffers,
dtype promotion, and shape-mismatch error handling.
"""

import numpy as np
from gpt2_triton.kernels.add import add


def test_add_correctness():
    """Test numerical correctness vs NumPy at multiple sizes."""
    print("\n=== Add Correctness Tests ===")

    test_sizes = [
        128,
        1024,
        777,   # non-power-of-2
        1,     # single element
    ]

    np.random.seed(42)
    for size in test_sizes:
        x = np.random.randn(size).astype(np.float32)
        y = np.random.randn(size).astype(np.float32)
        out = add(x, y)
        ref = x + y

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] N={size:>5} | max_diff={max_diff:.2e}")
        assert passed, f"Add failed for N={size}, max_diff={max_diff:.2e}"


def test_add_edge_cases():
    """Stability with extreme values and uniform arrays."""
    print("\n=== Add Edge Case Tests ===")

    # Large positive values (multi-block check)
    x_large = np.array([1e3, 1e6, 1e10], dtype=np.float32)
    y_large = np.array([2e3, 5e5, 2e10], dtype=np.float32)
    out_large = add(x_large, y_large)
    ref_large = x_large + y_large
    max_diff = float(np.abs(out_large - ref_large).max())
    passed = np.allclose(out_large, ref_large, atol=1e-3, rtol=1e-3)
    print(f"[{'PASS' if passed else 'FAIL'}] Large positive | max_diff={max_diff:.2e}")
    assert passed, f"Large positive add failed, max_diff={max_diff:.2e}"

    # Large negative values (multi-block check)
    x_neg = np.array([-1e3, -1e6, -1e10], dtype=np.float32)
    y_neg = np.array([-2e3, -5e5, -2e10], dtype=np.float32)
    out_neg = add(x_neg, y_neg)
    ref_neg = x_neg + y_neg
    max_diff = float(np.abs(out_neg - ref_neg).max())
    passed = np.allclose(out_neg, ref_neg, atol=1e-3, rtol=1e-3)
    print(f"[{'PASS' if passed else 'FAIL'}] Large negative | max_diff={max_diff:.2e}")
    assert passed, f"Large negative add failed, max_diff={max_diff:.2e}"

    # All zeros
    x_zero = np.zeros(128, dtype=np.float32)
    y_zero = np.zeros(128, dtype=np.float32)
    out_zero = add(x_zero, y_zero)
    ref_zero = x_zero + y_zero
    max_diff = float(np.abs(out_zero - ref_zero).max())
    passed = np.allclose(out_zero, ref_zero, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] All zeros    | max_diff={max_diff:.2e}")
    assert passed, f"All zeros add failed, max_diff={max_diff:.2e}"

    # All same value
    x_same = np.full(256, 3.14159, dtype=np.float32)
    y_same = np.full(256, 2.71828, dtype=np.float32)
    out_same = add(x_same, y_same)
    ref_same = x_same + y_same
    max_diff = float(np.abs(out_same - ref_same).max())
    passed = np.allclose(out_same, ref_same, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Same values  | max_diff={max_diff:.2e}")
    assert passed, f"Same values add failed, max_diff={max_diff:.2e}"


def test_add_empty_array():
    """Empty input (N=0) should return an empty float32 array."""
    print("\n=== Add Empty Array Test ===")

    # 1D empty arrays
    x = np.array([], dtype=np.float32)
    y = np.array([], dtype=np.float32)
    out = add(x, y)
    assert out.shape == (0,), f"Expected shape (0,), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"

    # Multi-dimensional empty arrays
    x_multi = np.empty((0, 10, 5), dtype=np.float32)
    y_multi = np.empty((0, 10, 5), dtype=np.float32)
    out_multi = add(x_multi, y_multi)
    assert out_multi.shape == (0, 10, 5), f"Expected shape (0, 10, 5), got {out_multi.shape}"
    assert out_multi.dtype == np.float32, f"Expected float32, got {out_multi.dtype}"

    print("[PASS] Empty arrays handled correctly")


def test_add_multidim():
    """Multi-dimensional input should preserve shape."""
    print("\n=== Add Multi-dimensional Test ===")
    np.random.seed(7)
    shape = (4, 16, 128)
    x = np.random.randn(*shape).astype(np.float32)
    y = np.random.randn(*shape).astype(np.float32)
    out = add(x, y)
    ref = x + y
    assert out.shape == shape, f"Shape mismatch: {out.shape} vs {shape}"
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] shape={shape} | max_diff={max_diff:.2e}")
    assert passed, f"Multi-dim add failed, max_diff={max_diff:.2e}"


def test_add_non_contiguous():
    """Non-contiguous (strided) input should be handled via ascontiguousarray."""
    print("\n=== Add Non-contiguous Input Test ===")

    # Create 2D matrix, take a column (strided slice) for both x and y.
    matrix_x = np.random.randn(8, 16).astype(np.float32)
    matrix_y = np.random.randn(8, 16).astype(np.float32)
    col_x = matrix_x[:, 3]  # non-contiguous column slice
    col_y = matrix_y[:, 3]
    assert not col_x.flags["C_CONTIGUOUS"], "Test precondition failed: column slice should be non-contiguous"
    out = add(col_x, col_y)
    ref = col_x + col_y
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Strided column  | max_diff={max_diff:.2e}")
    assert passed, f"Non-contiguous add failed, max_diff={max_diff:.2e}"

    # Also test a row slice (contiguous, but good for completeness).
    row_x = matrix_x[3, :]
    row_y = matrix_y[3, :]
    out_row = add(row_x, row_y)
    ref_row = row_x + row_y
    passed_row = np.allclose(out_row, ref_row, atol=1e-4)
    print(f"[{'PASS' if passed_row else 'FAIL'}] Row slice       | max_diff={float(np.abs(out_row - ref_row).max()):.2e}")
    assert passed_row, f"Row slice add failed, max_diff={float(np.abs(out_row - ref_row).max()):.2e}"


def test_add_dtype_conversion():
    """Integer and float64 inputs should be converted to float32 without error."""
    print("\n=== Add Dtype Conversion Test ===")

    # int32 input.
    x_int = np.array([-2, 0, 1, 3], dtype=np.int32)
    y_int = np.array([4, -1, 2, 5], dtype=np.int32)
    out_int = add(x_int, y_int)
    assert out_int.dtype == np.float32, f"Expected float32, got {out_int.dtype}"
    ref_int = x_int.astype(np.float32) + y_int.astype(np.float32)
    assert np.allclose(out_int, ref_int, atol=1e-4), "int32 add failed"

    # float64 input.
    x_f64 = np.array([-1.5, 0.0, 2.0, 3.14], dtype=np.float64)
    y_f64 = np.array([2.5, 1.0, -1.0, 0.86], dtype=np.float64)
    out_f64 = add(x_f64, y_f64)
    assert out_f64.dtype == np.float32, f"Expected float32, got {out_f64.dtype}"
    ref_f64 = x_f64.astype(np.float32) + y_f64.astype(np.float32)
    assert np.allclose(out_f64, ref_f64, atol=1e-4), "float64 add failed"

    print("[PASS] int32 and float64 inputs correctly converted to float32")


def test_add_shape_mismatch():
    """Unequal shapes should raise AssertionError."""
    print("\n=== Add Shape Mismatch Test ===")

    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.array([1.0, 2.0], dtype=np.float32)

    try:
        add(x, y)
        raise RuntimeError("Expected AssertionError for shape mismatch")
    except AssertionError:
        print("[PASS] Shape mismatch correctly raises AssertionError")


if __name__ == "__main__":
    print("Running Add unit tests on current GPU backend...")
    test_add_correctness()
    test_add_edge_cases()
    test_add_empty_array()
    test_add_multidim()
    test_add_non_contiguous()
    test_add_dtype_conversion()
    test_add_shape_mismatch()
    print("\n" + "=" * 45)
    print("All Add tests PASSED")
    print("=" * 45)
