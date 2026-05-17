"""
Minimal working GEMM demo using pure Python + Triton (no PyTorch)

This script demonstrates:
- GPU memory management via ctypes
- Triton GEMM kernel execution
- Correctness and performance numbers
"""

import sys
import time
import numpy as np
sys.path.insert(0, ".")
from gpt2_triton.kernels.gemm import gemm

def main():
    print("=" * 50)
    print("Triton GEMM Demo (Pure Python + Triton, no PyTorch)")
    print("=" * 50)

    # Test sizes
    M, K, N = 256, 512, 256
    print(f"\nMatrix sizes: {M}x{K} @ {K}x{N}")

    np.random.seed(42)
    a = np.random.randn(M, K).astype(np.float32)
    b = np.random.randn(K, N).astype(np.float32)

    # Reference
    ref = a @ b

    # Triton version
    t0 = time.perf_counter()
    c = gemm(a, b)
    t1 = time.perf_counter()

    # Metrics
    max_diff = np.abs(c - ref).max()
    avg_time_ms = (t1 - t0) * 1000

    print(f"\nResults:")
    print(f"  Max absolute difference : {max_diff:.2e}")
    print(f"  Kernel time             : {avg_time_ms:.2f} ms")
    print(f"  Output shape            : {c.shape}")

    if max_diff < 0.1:
        print("\n✅ Demo successful - kernel runs correctly on GPU")
    else:
        print("\n⚠️  Demo ran but accuracy needs improvement")

if __name__ == "__main__":
    main()