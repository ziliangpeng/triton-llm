"""
Unit tests for Triton Softmax kernel.

Tests correctness against a pure NumPy reference implementation at various
sizes and edge cases, including numerical stability with extreme values.
"""

import numpy as np
from gpt2_triton.kernels.softmax import softmax


def _softmax_ref(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Pure NumPy numerically stable softmax (no scipy dependency)."""
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def test_softmax_correctness():
    """Test numerical correctness vs NumPy reference at multiple shapes."""
    print("\n=== Softmax Correctness Tests ===")

    test_shapes = [
        (1, 128),
        (4, 256),
        (8, 777),
        (1, 1),
        (4, 2048),
    ]

    np.random.seed(42)
    for shape in test_shapes:
        x = np.random.randn(*shape).astype(np.float32)
        out = softmax(x)
        ref = _softmax_ref(x)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] shape={str(shape):>12} | max_diff={max_diff:.2e}")
        assert passed, f"Softmax failed for shape={shape}, max_diff={max_diff:.2e}"


def test_softmax_numerical_stability():
    """Verify softmax is stable with extreme values (no NaN/Inf)."""
    print("\n=== Softmax Numerical Stability Tests ===")

    test_cases = [
        ("Large positives", np.array([[100.0, 1000.0, 1e6]], dtype=np.float32)),
        ("Large negatives", np.array([[-100.0, -1000.0, -1e6]], dtype=np.float32)),
        ("Mixed signs", np.array([[1e6, -1e6, 0.0]], dtype=np.float32)),
        ("Wide range", np.random.randn(4, 512).astype(np.float32) * 100),
    ]

    all_passed = True
    for name, x in test_cases:
        out = softmax(x)
        ref = _softmax_ref(x)

        has_nan = np.any(np.isnan(out))
        has_inf = np.any(np.isinf(out))
        sums = out.sum(axis=1)
        sums_close = np.allclose(sums, np.ones(sums.shape), atol=1e-4)
        max_diff = float(np.abs(out - ref).max())

        passed = (not has_nan) and (not has_inf) and sums_close and (max_diff < 1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name:<20} | NaN={has_nan} Inf={has_inf} sum_close={sums_close} max_diff={max_diff:.2e}")
        assert passed, f"Numerical stability failed for {name}"


def test_softmax_uniform_input():
    """All-equal input → each output = 1/N."""
    print("\n=== Softmax Uniform Input Test ===")

    all_passed = True
    for N in [4, 16, 128, 777]:
        x = np.full((2, N), 2.5, dtype=np.float32)
        out = softmax(x)
        expected = np.full((2, N), 1.0 / N, dtype=np.float32)
        max_diff = float(np.abs(out - expected).max())
        passed = np.allclose(out, expected, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] N={N:>4} | max_diff={max_diff:.2e}")
        assert passed, f"Uniform input failed for N={N}, max_diff={max_diff:.2e}"


def test_softmax_1d_input():
    """1D array should be handled correctly (flattened output)."""
    print("\n=== Softmax 1D Input Test ===")

    N = 256
    x = np.random.randn(N).astype(np.float32)
    out = softmax(x)

    assert out.ndim == 1, f"Expected 1D output, got {out.ndim}D"
    assert out.shape == (N,), f"Expected shape ({N},), got {out.shape}"

    ref = _softmax_ref(x.reshape(1, -1)).ravel()
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] N={N} | max_diff={max_diff:.2e}")
    assert passed, f"1D softmax failed, max_diff={max_diff:.2e}"

    return passed


def test_softmax_empty_array():
    """Empty input should return an empty float32 array."""
    print("\n=== Softmax Empty Array Test ===")

    # (0, 10) — no rows
    x = np.empty((0, 10), dtype=np.float32)
    out = softmax(x)
    assert out.shape == (0, 10), f"Expected shape (0, 10), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    print(f"[PASS] shape (0, 10)")

    # (4, 0) — no columns
    x = np.empty((4, 0), dtype=np.float32)
    out = softmax(x)
    assert out.shape == (4, 0), f"Expected shape (4, 0), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    print(f"[PASS] shape (4, 0)")

    # 1D empty array
    x = np.array([], dtype=np.float32)
    out = softmax(x)
    assert out.shape == (0,), f"Expected shape (0,), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"
    print(f"[PASS] shape (0,)")

    print("[PASS] All empty array tests passed")


def test_softmax_dtype_conversion():
    """Integer and float64 inputs should be converted to float32."""
    print("\n=== Softmax Dtype Conversion Test ===")

    # int32 input
    x_int = np.array([[-2, 0, 1, 3]], dtype=np.int32)
    out_int = softmax(x_int)
    assert out_int.dtype == np.float32, f"Expected float32, got {out_int.dtype}"
    ref_int = _softmax_ref(x_int.astype(np.float32))
    assert np.allclose(out_int, ref_int, atol=1e-4), "int32 softmax failed"
    print("[PASS] int32 input")

    # float64 input
    x_f64 = np.array([[-1.5, 0.0, 2.0]], dtype=np.float64)
    out_f64 = softmax(x_f64)
    assert out_f64.dtype == np.float32, f"Expected float32, got {out_f64.dtype}"
    ref_f64 = _softmax_ref(x_f64.astype(np.float32))
    assert np.allclose(out_f64, ref_f64, atol=1e-4), "float64 softmax failed"
    print("[PASS] float64 input")

    print("[PASS] All dtype conversion tests passed")


def test_softmax_non_contiguous():
    """Non-contiguous (strided) input should be handled correctly."""
    print("\n=== Softmax Non-contiguous Input Test ===")

    # Create 2D matrix, take every-other row (strided).
    matrix = np.random.randn(8, 128).astype(np.float32)
    strided = matrix[::2, :]  # every other row — non-contiguous in rows
    assert not strided.flags["C_CONTIGUOUS"], "Test precondition: strided slice should be non-contiguous"

    out = softmax(strided)
    ref = _softmax_ref(strided)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Strided rows  | max_diff={max_diff:.2e}")
    assert passed, f"Non-contiguous softmax failed, max_diff={max_diff:.2e}"

    print("[PASS] Non-contiguous input test passed")


def test_softmax_axis_validation():
    """Axis-related validation tests."""
    print("\n=== Softmax Axis Validation Test ===\n")

    # 1D input: axis=0 is valid (the only axis)
    x_1d = np.random.randn(256).astype(np.float32)
    out_1d = softmax(x_1d, axis=0)
    ref_1d = softmax(x_1d)  # default axis=-1, same as axis=0 for 1D
    assert np.allclose(out_1d, ref_1d, atol=1e-6), "1D axis=0 should work"
    print("[PASS] 1D input with axis=0 works")

    # 2D input: axis=-1 and axis=1 should produce same result
    x_2d = np.random.randn(4, 128).astype(np.float32)
    out1 = softmax(x_2d, axis=-1)
    out2 = softmax(x_2d, axis=1)
    assert np.allclose(out1, out2, atol=1e-6), "axis=-1 and axis=1 should produce same result"
    print("[PASS] axis=-1 and axis=1 produce same result")

    # axis=0 should raise on 2D input
    try:
        softmax(x_2d, axis=0)
        raise RuntimeError("Expected NotImplementedError for axis=0 on 2D")
    except NotImplementedError:
        print("[PASS] axis=0 on 2D raises NotImplementedError")

    # axis=-2 should raise on 2D input (equivalent to axis=0)
    try:
        softmax(x_2d, axis=-2)
        raise RuntimeError("Expected NotImplementedError for axis=-2")
    except NotImplementedError:
        print("[PASS] axis=-2 on 2D raises NotImplementedError")

    print("\n[PASS] All axis validation tests passed")


if __name__ == "__main__":
    print("Running Softmax unit tests on current GPU backend...")
    test_softmax_correctness()
    test_softmax_numerical_stability()
    test_softmax_uniform_input()
    test_softmax_1d_input()
    test_softmax_empty_array()
    test_softmax_dtype_conversion()
    test_softmax_non_contiguous()
    test_softmax_axis_validation()
    print("\n" + "=" * 45)
    print("All Softmax tests PASSED")
    print("=" * 45)
