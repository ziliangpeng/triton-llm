"""
Comprehensive test for Triton GEMM kernel
- Correctness vs numpy
- Performance on H100
"""

import sys
import time
import numpy as np
sys.path.insert(0, ".")
from gpt2_triton.kernels.gemm import gemm

def test_gemm_correctness(M, K, N, runs=10):
    print(f"\n=== Testing GEMM: {M}x{K} @ {K}x{N} ===")
    np.random.seed(42)
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Reference
    ref = a @ b

    # Triton version
    c = gemm(a, b)

    max_diff = np.abs(c - ref).max()
    rel_diff = max_diff / (np.abs(ref).max() + 1e-8)
    print(f"Max absolute diff : {max_diff:.6e}")
    print(f"Max relative diff : {rel_diff:.6e}")
    passed = max_diff < 1e-2
    print("Correctness:", "PASS" if passed else "FAIL")
    return passed

def benchmark_gemm(M, K, N, warmup=5, runs=20):
    print(f"\n=== Benchmark GEMM: {M}x{K} @ {K}x{N} ===")
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = gemm(a, b)

    # Benchmark
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = gemm(a, b)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = np.mean(times) * 1000  # ms
    min_time = np.min(times) * 1000
    print(f"Average time: {avg_time:.3f} ms")
    print(f"Min time    : {min_time:.3f} ms")
    return avg_time

if __name__ == "__main__":
    sizes = [
        (64, 128, 32),
        (128, 256, 128),
        (512, 512, 512),
    ]

    print("=== Triton GEMM Comprehensive Test ===\n")
    all_passed = True
    for M, K, N in sizes:
        passed = test_gemm_correctness(M, K, N)
        all_passed &= passed
        benchmark_gemm(M, K, N)

    print("\n=== Final Result ===")
    print("All tests passed!" if all_passed else "Some tests FAILED")
