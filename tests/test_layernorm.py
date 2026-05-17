"""
Unit tests for Triton LayerNorm kernel.
"""

import sys
import time
import numpy as np

sys.path.insert(0, "/home/ziliang/work/triton-llm")

from gpt2_triton.kernels.layernorm import layernorm


def test_layernorm_correctness():
    """Test numerical correctness against numpy reference."""
    print("\n=== LayerNorm Correctness Tests ===")

    test_cases = [
        (4, 128),
        (8, 256),
        (16, 768),
    ]

    all_passed = True

    for batch, hidden in test_cases:
        np.random.seed(42)
        x = np.random.randn(batch, hidden).astype(np.float32)
        weight = np.ones(hidden, dtype=np.float32)
        bias = np.zeros(hidden, dtype=np.float32)

        # Reference
        mean = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        ref = (x - mean) / np.sqrt(var + 1e-5) * weight + bias

        # Triton
        out = layernorm(x, weight, bias)

        max_diff = np.abs(out - ref).max()
        passed = max_diff < 1e-2
        status = "PASS" if passed else "FAIL"

        print(f"[{status}] batch={batch}, hidden={hidden} | max_diff={max_diff:.2e}")
        all_passed &= passed

    return all_passed


def test_layernorm_performance():
    """Basic performance test."""
    print("\n=== LayerNorm Performance Test ===")

    batch, hidden = 32, 1024
    x = np.random.randn(batch, hidden).astype(np.float32)
    weight = np.ones(hidden, dtype=np.float32)
    bias = np.zeros(hidden, dtype=np.float32)

    # Warmup
    for _ in range(3):
        _ = layernorm(x, weight, bias)

    # Benchmark
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = layernorm(x, weight, bias)
        times.append(time.perf_counter() - t0)

    avg_ms = np.mean(times) * 1000
    print(f"Size: batch={batch}, hidden={hidden}")
    print(f"Avg time: {avg_ms:.2f} ms")

    return avg_ms


if __name__ == "__main__":
    print("Running LayerNorm unit tests...")
    correctness = test_layernorm_correctness()
    perf = test_layernorm_performance()

    print("\n" + "=" * 40)
    print("All tests PASSED" if correctness else "Some tests FAILED")
    print("=" * 40)