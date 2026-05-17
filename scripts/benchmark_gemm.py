"""
GEMM Benchmark Script
Compares Triton GEMM vs NumPy vs cuBLAS/hipBLAS (via PyTorch)
"""

import time
import numpy as np
import torch
from gpt2_triton.kernels.gemm import gemm as triton_gemm


def benchmark(func, a, b, warmup=5, repeat=20):
    """Run benchmark with warmup and return average time in ms."""
    for _ in range(warmup):
        _ = func(a, b)

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = func(a, b)
        times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000  # ms


def run_benchmarks(sizes):
    results = []

    for M, K, N in sizes:
        print(f"\n=== Benchmarking {M}x{K} @ {K}x{N} ===")

        a_np = np.random.randn(M, K).astype(np.float32)
        b_np = np.random.randn(K, N).astype(np.float32)

        # NumPy
        t_np = benchmark(lambda x, y: x @ y, a_np, b_np)
        print(f"NumPy:        {t_np:.2f} ms")

        # Triton
        t_triton = benchmark(triton_gemm, a_np, b_np)
        print(f"Triton:       {t_triton:.2f} ms")

        # cuBLAS / hipBLAS via PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        a_torch = torch.from_numpy(a_np).to(device)
        b_torch = torch.from_numpy(b_np).to(device)

        # Warmup
        for _ in range(5):
            _ = a_torch @ b_torch
        torch.cuda.synchronize() if device == "cuda" else None

        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = a_torch @ b_torch
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        t_blas = np.mean(times) * 1000
        print(f"cuBLAS/hipBLAS: {t_blas:.2f} ms")

        results.append({
            "size": f"{M}x{K}@{K}x{N}",
            "numpy_ms": t_np,
            "triton_ms": t_triton,
            "blas_ms": t_blas
        })

    return results


if __name__ == "__main__":
    sizes = [
        (512, 1024, 512),
        (1024, 2048, 1024),
        (2048, 4096, 2048),
    ]

    results = run_benchmarks(sizes)

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['size']:20s} | NumPy: {r['numpy_ms']:.1f}ms | Triton: {r['triton_ms']:.1f}ms | BLAS: {r['blas_ms']:.1f}ms")