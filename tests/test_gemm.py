"""
Unit tests for Triton GEMM kernel (CUDA + HIP).
"""

import numpy as np
import time
from gpt2_triton.kernels.gemm import gemm


def test_gemm_correctness():
    """Test numerical correctness against numpy matmul."""
    print("\n=== GEMM Correctness Tests ===")

    test_cases = [
        (64, 128, 32),
        (128, 256, 128),
        (256, 512, 256),
    ]

    all_passed = True
    for M, K, N in test_cases:
        np.random.seed(42)
        a = np.random.randn(M, K).astype(np.float32)
        b = np.random.randn(K, N).astype(np.float32)

        ref = a @ b
        out = gemm(a, b)

        max_diff = np.abs(out - ref).max()
        passed = max_diff < 0.1
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] {M}x{K} @ {K}x{N} | max_diff={max_diff:.2e}")
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

    return avg_ms


if __name__ == "__main__":
    print("Running GEMM unit tests on current GPU backend...\n")
    correctness = test_gemm_correctness()
    perf = test_gemm_performance()

    print("\n" + "=" * 45)
    print("All GEMM tests PASSED" if correctness else "Some tests FAILED")
    print("=" * 45)