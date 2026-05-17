"""
Unit tests for the GPU Memory Allocator.
"""

import numpy as np
from gpt2_triton.gpu import allocate, to_device, to_host, DeviceTensor


def test_allocate_and_roundtrip():
    """Test basic allocation and host <-> device roundtrip."""
    shape = (4, 8)
    arr = np.random.randn(*shape).astype(np.float32)

    dev = to_device(arr)
    back = to_host(dev)

    assert isinstance(dev, DeviceTensor)
    assert back.shape == shape
    assert np.allclose(arr, back), "Roundtrip data mismatch"


def test_allocate_different_dtypes():
    """Test allocation with different numpy dtypes."""
    for dtype in [np.float32, np.float64]:
        arr = np.ones((3, 3), dtype=dtype)
        dev = to_device(arr)
        back = to_host(dev)
        assert back.dtype == dtype


if __name__ == "__main__":
    test_allocate_and_roundtrip()
    test_allocate_different_dtypes()
    print("All GPU allocator tests passed!")