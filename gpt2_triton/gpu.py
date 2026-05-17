"""
GPU memory management using Triton's own driver (recommended way).
This returns proper pointer objects that Triton kernels can use directly.
"""

import numpy as np
import triton
from triton.runtime.driver import driver

def get_device():
    return driver.get_current_device()

def allocate(shape, dtype=np.float32):
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    device = get_device()
    allocator = driver.get_allocator(device)
    ptr = allocator.allocate(nbytes)
    return DeviceTensor(ptr, shape, dtype, nbytes)

class DeviceTensor:
    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr          # This should be a proper Triton pointer object
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def to_numpy(self):
        host = np.empty(self.shape, dtype=self.dtype)
        # Use Triton's copy or cudaMemcpy
        # For simplicity, we'll implement via ctypes for now
        import ctypes
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.ptr.value if hasattr(self.ptr, 'value') else self.ptr),
            self.nbytes,
            ctypes.c_int(2)
        )
        return host

    def from_numpy(self, arr: np.ndarray):
        import ctypes
        cudart = ctypes.CDLL('libcudart.so')
        cudart.cudaMemcpy(
            ctypes.c_void_p(self.ptr.value if hasattr(self.ptr, 'value') else self.ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            self.nbytes,
            ctypes.c_int(1)
        )

    def data_ptr(self):
        # Return the pointer in the format Triton expects
        if hasattr(self.ptr, 'value'):
            return self.ptr.value
        return self.ptr

def to_device(arr: np.ndarray):
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor

def to_host(tensor: DeviceTensor):
    return tensor.to_numpy()