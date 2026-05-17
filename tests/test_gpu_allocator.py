"""
Basic tests for the GPU allocator.
"""

import numpy as np
from gpt2_triton.gpu import to_device, to_host


def test_roundtrip():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    dev = to_device(arr)
    back = to_host(dev)
    assert np.allclose(arr, back)


def test_2d_array():
    arr = np.random.randn(4, 8).astype(np.float32)
    dev = to_device(arr)
    back = to_host(dev)
    assert back.shape == arr.shape
    assert np.allclose(arr, back)


if __name__ == "__main__":
    test_roundtrip()
    test_2d_array()
    print("All allocator tests passed!")