"""
Unit tests for Triton GEMM kernel (CUDA + HIP).
"""

import numpy as np
import time
from triton_llm.kernels.gemm import gemm


def test_gemm_correctness():
    """Test numerical correctness against numpy matmul using np.allclose."""
    print("\n=== GEMM Correctness Tests ===")

    test_cases = [
        (64, 128, 32),
        (128, 256, 128),
        (256, 512, 256),
        # Non-block-aligned cases to test boundary masking
        (65, 130, 33),
        (100, 200, 150),
    ]

    all_passed = True
    for M, K, N in test_cases:
        np.random.seed(42)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        ref = a @ b
        out = gemm(a, b)

        # Use np.allclose with reasonable tolerance
        passed = np.allclose(out, ref, rtol=1e-2, atol=5e-2)
        max_diff = np.abs(out - ref).max()

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {M}x{K} @ {K}x{N} | max_diff={max_diff:.2e} | allclose={passed}")

        assert passed, f"GEMM failed for {M}x{K}@{K}x{N}, max_diff={max_diff:.2e}"

        all_passed &= passed

    return all_passed


def test_gemm_performance():
    """Basic performance benchmark."""
    print("\n=== GEMM Performance Test ===")

    M, K, N = 512, 1024, 512
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(3):
        _ = gemm(a, b)

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = gemm(a, b)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    min_ms = np.min(times) * 1000

    print(f"Size: {M}x{K} @ {K}x{N}")
    print(f"Avg time: {avg_ms:.2f} ms")
    print(f"Min time: {min_ms:.2f} ms")

    # Loose upper bound for CI
    assert avg_ms < 50, f"GEMM too slow: {avg_ms:.2f}ms"

    return avg_ms


def test_gemm_edge_cases():
    """Test edge cases."""
    print("\n=== GEMM Edge Cases ===")

    # K=0 should return a zero matrix (gemm handles it explicitly now)
    a = np.random.randn(10, 0).astype(np.float32)
    b = np.random.randn(0, 10).astype(np.float32)
    out = gemm(a, b)
    assert out.shape == (10, 10), f"Expected shape (10, 10), got {out.shape}"
    assert np.all(out == 0.0), "K=0 result should be all zeros"
    print("[PASS] K=0 case handled (returned zero matrix)")

    # M=0 should return a zero matrix
    a = np.random.randn(0, 20).astype(np.float32)
    b = np.random.randn(20, 10).astype(np.float32)
    out = gemm(a, b)
    assert out.shape == (0, 10), f"Expected shape (0, 10), got {out.shape}"
    assert out.size == 0, "M=0 result should be empty"
    print("[PASS] M=0 case handled (returned zero matrix)")

    # N=0 should return a zero matrix
    a = np.random.randn(10, 20).astype(np.float32)
    b = np.random.randn(20, 0).astype(np.float32)
    out = gemm(a, b)
    assert out.shape == (10, 0), f"Expected shape (10, 0), got {out.shape}"
    assert out.size == 0, "N=0 result should be empty"
    print("[PASS] N=0 case handled (returned zero matrix)")

    # Mismatched K should raise AssertionError
    a = np.random.randn(10, 20).astype(np.float32)
    b = np.random.randn(30, 10).astype(np.float32)
    try:
        _ = gemm(a, b)
        raise RuntimeError("Should have raised AssertionError")
    except AssertionError:
        print("[PASS] Mismatched K correctly rejected")

    return True


if __name__ == "__main__":
    print("Running GEMM unit tests on current GPU backend...\n")
    correctness = test_gemm_correctness()
    perf = test_gemm_performance()
    edge_cases = test_gemm_edge_cases()

    print("\n" + "=" * 45)
    if correctness and edge_cases:
        print("All GEMM tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 45)
