"""
Unit tests for Triton GEMM kernel.

Reference implementation uses numpy for correctness checking.
"""

import sys
import time
import numpy as np
sys.path.insert(0, "..")

from gpt2_triton.kernels.gemm import gemm


def test_gemm_correctness():
    """Test numerical correctness against numpy reference."""
    test_cases = [
        (64, 128, 32),
        (128, 256, 128),
        (256, 512, 256),
    ]

    print("\n=== GEMM Correctness Tests ===")
    all_passed = True

    for M, K, N in test_cases:
        np.random.seed(42)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        ref = a @ b
        out = gemm(a, b)

        max_abs_diff = np.abs(out - ref).max()
        rel_diff = max_abs_diff / (np.abs(ref).max() + 1e-8)

        passed = max_abs_diff < 0.1
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {M}x{K} @ {K}x{N} | max_diff={max_abs_diff:.2e} | rel={rel_diff:.2e}")
        all_passed &= passed

    return all_passed


def test_gemm_performance():
    """Basic performance test."""
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

    return avg_ms


if __name__ == "__main__":
    print("Running GEMM unit tests...\n")
    correctness = test_gemm_correctness()
    perf = test_gemm_performance()

    print("\n" + "=" * 40)
    if correctness:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 40)