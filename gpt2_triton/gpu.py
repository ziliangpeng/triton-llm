"""
GPU memory management using Triton's driver (Triton 3.1.0 compatible).
"""

import numpy as np
from triton.runtime.driver import driver

def get_device():
    return driver.active

def allocate(shape, dtype=np.float32):
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    dev = get_device()
    allocator = dev.allocator
    ptr = allocator.allocate(nbytes)
    return DeviceTensor(ptr, shape, dtype, nbytes)

class DeviceTensor:
    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def to_numpy(self):
        import ctypes
        host = np.empty(self.shape, dtype=self.dtype)
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.ptr),
            self.nbytes,
            ctypes.c_int(2)
        )
        return host

    def from_numpy(self, arr: np.ndarray):
        import ctypes
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(self.ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            self.nbytes,
            ctypes.c_int(1)
        )

    def data_ptr(self):
        return self.ptr

def to_device(arr: np.ndarray):
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor

def to_host(tensor: DeviceTensor):
    return tensor.to_numpy()