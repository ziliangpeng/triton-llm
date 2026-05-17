"""
Tests for the GPU Memory Allocator (CUDA and HIP backends).
"""

import numpy as np
from gpt2_triton.gpu import to_device, to_host, allocate


def test_roundtrip():
    """Test basic allocation and host <-> device roundtrip."""
    arr = np.random.randn(4, 8).astype(np.float32)
    dev = to_device(arr)
    back = to_host(dev)

    assert back.shape == arr.shape
    assert np.allclose(arr, back)


def test_different_dtypes():
    """Test support for common dtypes."""
    for dtype in [np.float32, np.float64]:
        arr = np.ones((3, 3), dtype=dtype)
        dev = to_device(arr)
        back = to_host(dev)
        assert back.dtype == dtype
        assert np.allclose(arr, back)


def test_memory_cleanup():
    """Ensure DeviceTensor can be deleted without error."""
    arr = np.ones((10,), dtype=np.float32)
    dev = to_device(arr)
    del dev  # Should trigger __del__ without raising


if __name__ == "__main__":
    test_roundtrip()
    test_different_dtypes()
    test_memory_cleanup()
    print("All GPU allocator tests passed!")