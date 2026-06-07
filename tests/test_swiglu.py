"""
Unit tests for Triton SwiGLU kernel (CUDA + HIP).

Tests correctness, edge cases, empty input, multi-dimensional
preservation, and shape-mismatch validation.
"""

import numpy as np
from smollm2_triton.kernels.swiglu import swiglu


def silu_ref(x):
    """NumPy reference SiLU: x * sigmoid(x) with numerically stable branch."""
    # For x >= 0: use 1/(1+exp(-x)) — exp(-x) is safe, ≤ 1
    # For x < 0:  use exp(x)/(1+exp(x)) — exp(x) is safe, < 1
    sigmoid = np.where(x >= 0,
                       1.0 / (1.0 + np.exp(-x)),
                       np.exp(x) / (1.0 + np.exp(x)))
    return x * sigmoid


def swiglu_ref(gate, up):
    """NumPy reference SwiGLU."""
    return silu_ref(gate) * up


def test_swiglu_correctness():
    """Correctness vs numpy reference across several sizes."""
    print("\n=== SwiGLU Correctness Tests ===")

    np.random.seed(42)
    test_sizes = [
        1536,      # Non-power-of-2
        2560,      # Non-power-of-2
        8192,      # Power-of-2
        777,       # Small non-power-of-2
    ]

    for N in test_sizes:
        gate = np.random.randn(N).astype(np.float32)
        up = np.random.randn(N).astype(np.float32)

        out = swiglu(gate, up)
        ref = swiglu_ref(gate, up)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] N={N:>5} | max_diff={max_diff:.2e}")
        assert passed, f"SwiGLU failed for N={N}, max_diff={max_diff:.2e}"


def test_swiglu_edge_cases():
    """Test edge cases: large values, all zeros, single element."""
    print("\n=== SwiGLU Edge Case Tests ===")

    # Large positive values (clamped to 20).
    gate = np.array([100.0, 500.0, 1e10], dtype=np.float32)
    up = np.ones_like(gate)
    out = swiglu(gate, up)
    ref = swiglu_ref(gate, up)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Large positive | max_diff={max_diff:.2e}")
    assert passed, f"Large-positive SwiGLU failed, max_diff={max_diff:.2e}"

    # Large negative values (clamped to -20).
    gate = np.array([-100.0, -500.0, -1e10], dtype=np.float32)
    up = np.ones_like(gate)
    out = swiglu(gate, up)
    ref = swiglu_ref(gate, up)
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Large negative | max_diff={max_diff:.2e}")
    assert passed, f"Large-negative SwiGLU failed, max_diff={max_diff:.2e}"

    # All zeros.
    gate = np.zeros((16, 128), dtype=np.float32)
    up = np.ones_like(gate)
    out = swiglu(gate, up)
    ref = swiglu_ref(gate, up)
    zero_passed = np.all(out == 0.0)
    max_diff = float(np.abs(out - ref).max())
    print(f"[{'PASS' if zero_passed else 'FAIL'}] All zeros    | max_diff={max_diff:.2e} | all_zero={zero_passed}")
    assert zero_passed, f"All-zero SwiGLU failed, max_diff={max_diff:.2e}"

    # Single element.
    gate = np.array([3.0], dtype=np.float32)
    up = np.array([2.0], dtype=np.float32)
    out = swiglu(gate, up)
    ref = swiglu_ref(gate, up)
    # Manual: silu(3) ≈ 3 / (1 + e⁻³) ≈ 3 / 1.0498 ≈ 2.857, * 2 ≈ 5.714
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] Single element | max_diff={max_diff:.2e} | val={out.item():.4f}")
    assert passed, f"Single-element SwiGLU failed, max_diff={max_diff:.2e}"


def test_swiglu_empty():
    """N=0 should return an empty array with the correct shape."""
    print("\n=== SwiGLU Empty Input Test ===")
    gate = np.empty((0, 1536), dtype=np.float32)
    up = np.empty((0, 1536), dtype=np.float32)

    out = swiglu(gate, up)
    passed = out.shape == (0, 1536) and out.dtype == np.float32 and out.size == 0
    print(f"[{'PASS' if passed else 'FAIL'}] Empty input | shape={out.shape}")
    assert passed, f"Empty input failed, shape={out.shape}"


def test_swiglu_multidim():
    """Multi-dimensional input preserves shape via leading-dim flattening."""
    print("\n=== SwiGLU Multi-dimensional Input Test ===")
    shape = (4, 16, 1536)
    np.random.seed(8)
    gate = np.random.randn(*shape).astype(np.float32)
    up = np.random.randn(*shape).astype(np.float32)

    out = swiglu(gate, up)
    ref = swiglu_ref(gate, up)

    shape_passed = out.shape == shape
    max_diff = float(np.abs(out - ref).max())
    value_passed = np.allclose(out, ref, atol=1e-4, rtol=1e-4)
    passed = shape_passed and value_passed
    print(f"[{'PASS' if passed else 'FAIL'}] shape={shape} | shape_ok={shape_passed} | max_diff={max_diff:.2e}")
    assert passed, f"Multi-dim SwiGLU failed, shape={out.shape}, max_diff={max_diff:.2e}"


def test_swiglu_mismatch():
    """Mismatched gate and up shapes should raise ValueError."""
    print("\n=== SwiGLU Shape Mismatch Validation ===")
    gate = np.random.randn(1536).astype(np.float32)
    up = np.random.randn(2560).astype(np.float32)

    try:
        swiglu(gate, up)
        assert False, "Should have rejected mismatched shapes"
    except ValueError:
        print("[PASS] Mismatched shapes correctly rejected")


if __name__ == "__main__":
    print("Running SwiGLU unit tests on current GPU backend...")
    test_swiglu_correctness()
    test_swiglu_edge_cases()
    test_swiglu_empty()
    test_swiglu_multidim()
    test_swiglu_mismatch()
    print("\n" + "=" * 45)
    print("All SwiGLU tests PASSED")
    print("=" * 45)
