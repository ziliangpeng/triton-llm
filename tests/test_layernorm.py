"""
Unit tests for Triton LayerNorm kernel (CUDA + HIP).
"""

import numpy as np
import time
from triton_llm.kernels.layernorm import layer_norm


def _layer_norm_reference(x, gamma, beta, eps=1e-5):
    """NumPy reference LayerNorm over the last dim."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return x_hat * gamma + beta


def test_layer_norm_correctness():
    """Correctness vs numpy reference across several shapes (incl. non-power-of-2 N)."""
    print("\n=== LayerNorm Correctness Tests ===")

    test_cases = [
        (8, 64),
        (32, 128),
        (64, 768),     # GPT-2 small hidden size
        (16, 1024),
        # Non-power-of-2 last dim: exercises the masked load path.
        (16, 100),
        (4, 33),
    ]

    np.random.seed(0)
    for M, N in test_cases:
        x = np.random.randn(M, N).astype(np.float32)
        gamma = np.random.randn(N).astype(np.float32)
        beta = np.random.randn(N).astype(np.float32)

        out = layer_norm(x, gamma, beta, eps=1e-5)
        ref = _layer_norm_reference(x, gamma, beta, eps=1e-5)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, rtol=1e-3, atol=1e-4)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] M={M:>4} N={N:>5} | max_diff={max_diff:.2e}")
        assert passed, f"LayerNorm failed for M={M}, N={N}, max_diff={max_diff:.2e}"


def test_layer_norm_identity_affine():
    """When gamma=1, beta=0, output should be zero-mean unit-variance per row."""
    print("\n=== LayerNorm Identity Affine Test ===")
    M, N = 32, 256
    np.random.seed(1)
    x = np.random.randn(M, N).astype(np.float32) * 10.0 + 5.0  # shifted/scaled
    gamma = np.ones(N, dtype=np.float32)
    beta = np.zeros(N, dtype=np.float32)

    out = layer_norm(x, gamma, beta, eps=1e-5)
    row_mean = out.mean(axis=-1)
    row_var = out.var(axis=-1)

    print(f"Row mean: max |mean|={np.abs(row_mean).max():.2e}")
    print(f"Row var : max |var-1|={np.abs(row_var - 1.0).max():.2e}")

    assert np.allclose(row_mean, 0.0, atol=1e-4), "Per-row mean is not ~0"
    assert np.allclose(row_var, 1.0, atol=1e-2), "Per-row var is not ~1"


def test_layer_norm_large_magnitude():
    """Numerical stability: large mean shouldn't blow up via E[x^2]-E[x]^2 cancellation."""
    print("\n=== LayerNorm Numerical Stability Test ===")
    M, N = 8, 512
    np.random.seed(2)
    # Add a large constant offset to stress the variance computation.
    x = (np.random.randn(M, N).astype(np.float32) + 1000.0)
    gamma = np.ones(N, dtype=np.float32)
    beta = np.zeros(N, dtype=np.float32)

    out = layer_norm(x, gamma, beta, eps=1e-5)
    ref = _layer_norm_reference(x, gamma, beta, eps=1e-5)

    max_diff = float(np.abs(out - ref).max())
    print(f"Large-offset (mean=1000) max_diff={max_diff:.2e}")
    assert np.allclose(out, ref, rtol=1e-2, atol=1e-2), (
        f"LayerNorm unstable for large-mean input, max_diff={max_diff:.2e}"
    )


def test_layer_norm_performance():
    """Basic performance benchmark at a GPT-2-sized shape."""
    print("\n=== LayerNorm Performance Test ===")
    M, N = 1024, 768
    np.random.seed(3)
    x = np.random.randn(M, N).astype(np.float32)
    gamma = np.random.randn(N).astype(np.float32)
    beta = np.random.randn(N).astype(np.float32)

    # Warmup.
    for _ in range(3):
        _ = layer_norm(x, gamma, beta)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = layer_norm(x, gamma, beta)
        times.append(time.perf_counter() - t0)

    avg_ms = float(np.mean(times) * 1000)
    min_ms = float(np.min(times) * 1000)
    print(f"Size: ({M}, {N}) | avg={avg_ms:.2f} ms | min={min_ms:.2f} ms")
    # Loose upper bound for CI — includes host<->device copies and allocation.
    assert avg_ms < 10, f"LayerNorm unexpectedly slow: {avg_ms:.2f} ms"


def test_layer_norm_input_validation():
    """Shape mismatches and invalid inputs should raise ValueError."""
    print("\n=== LayerNorm Input Validation ===")
    x = np.random.randn(8, 64).astype(np.float32)

    # Wrong gamma shape.
    try:
        layer_norm(x, np.ones(32, dtype=np.float32), np.zeros(64, dtype=np.float32))
        assert False, "Should have rejected mismatched gamma shape"
    except ValueError:
        print("[PASS] Mismatched gamma shape correctly rejected")

    # Wrong beta shape.
    try:
        layer_norm(x, np.ones(64, dtype=np.float32), np.zeros(32, dtype=np.float32))
        assert False, "Should have rejected mismatched beta shape"
    except ValueError:
        print("[PASS] Mismatched beta shape correctly rejected")

    # 1D input.
    try:
        layer_norm(np.random.randn(64).astype(np.float32),
                   np.ones(64, dtype=np.float32),
                   np.zeros(64, dtype=np.float32))
        assert False, "Should have rejected non-2D input"
    except ValueError:
        print("[PASS] Non-2D (1D) input correctly rejected")

    # N=0 last dimension.
    try:
        layer_norm(np.random.randn(4, 0).astype(np.float32),
                   np.ones(0, dtype=np.float32),
                   np.zeros(0, dtype=np.float32))
        assert False, "Should have rejected N=0"
    except ValueError:
        print("[PASS] N=0 correctly rejected")


def test_layer_norm_3d():
    """3D input (B, S, N) is supported via leading-dim flattening."""
    print("\n=== LayerNorm 3D Input Test ===")
    B, S, N = 4, 16, 128
    np.random.seed(7)
    x = np.random.randn(B, S, N).astype(np.float32)
    gamma = np.random.randn(N).astype(np.float32)
    beta = np.random.randn(N).astype(np.float32)

    out = layer_norm(x, gamma, beta)
    ref = _layer_norm_reference(x, gamma, beta)

    assert out.shape == (B, S, N), f"Output shape mismatch: {out.shape}"
    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, rtol=1e-3, atol=1e-4)
    print(f"[{'PASS' if passed else 'FAIL'}] shape=({B},{S},{N}) | max_diff={max_diff:.2e}")
    assert passed, f"3D LayerNorm failed, max_diff={max_diff:.2e}"


if __name__ == "__main__":
    print("Running LayerNorm unit tests on current GPU backend...")
    test_layer_norm_correctness()
    test_layer_norm_identity_affine()
    test_layer_norm_large_magnitude()
    test_layer_norm_performance()
    test_layer_norm_input_validation()
    test_layer_norm_3d()
    print("\n" + "=" * 45)
    print("All LayerNorm tests PASSED")
    print("=" * 45)
