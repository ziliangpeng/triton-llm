"""
Tests for the GPU Memory Allocator (CUDA and HIP backends).
"""

import numpy as np
from gpt2_triton.gpu import to_device, to_host, allocate


def test_backend_detected():
    """Basic smoke test to ensure allocator can be initialized."""
    # Just try to allocate something small
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dev = to_device(arr)
    back = to_host(dev)
    assert np.allclose(arr, back)


def test_roundtrip_1d():
    """1D array roundtrip between host and device."""
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dev = to_device(arr)
    back = to_host(dev)
    assert np.allclose(arr, back)


def test_roundtrip_2d():
    """2D array roundtrip between host and device."""
    arr = np.random.randn(4, 8).astype(np.float32)
    dev = to_device(arr)
    back = to_host(dev)
    assert back.shape == arr.shape
    assert np.allclose(arr, back)


def test_float64():
    """Test float64 dtype support."""
    arr = np.random.randn(3, 3).astype(np.float64)
    dev = to_device(arr)
    back = to_host(dev)
    assert back.dtype == np.float64
    assert np.allclose(arr, back)


def test_memory_freed():
    """Ensure __del__ runs without error."""
    arr = np.ones((10,), dtype=np.float32)
    dev = to_device(arr)
    del dev


if __name__ == "__main__":
    test_backend_detected()
    test_roundtrip_1d()
    test_roundtrip_2d()
    test_float64()
    test_memory_freed()
    print("All GPU allocator tests passed!")