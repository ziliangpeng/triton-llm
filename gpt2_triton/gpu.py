"""
Minimal GPU memory management using only Triton runtime + ctypes CUDA (no PyTorch).
"""

import numpy as np
import ctypes
import triton
from triton.runtime.driver import driver

# Load CUDA runtime for memcpy
cudart = ctypes.CDLL('libcudart.so')

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
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes
        self.itemsize = np.dtype(dtype).itemsize

    def to_numpy(self):
        host = np.empty(self.shape, dtype=self.dtype)
        # cudaMemcpy from device to host
        cudart.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_void_p(self.ptr),
            self.nbytes,
            ctypes.c_int(2)  # cudaMemcpyDeviceToHost
        )
        return host

    def from_numpy(self, arr: np.ndarray):
        cudart.cudaMemcpy(
            ctypes.c_void_p(self.ptr),
            arr.ctypes.data_as(ctypes.c_void_p),
            self.nbytes,
            ctypes.c_int(1)  # cudaMemcpyHostToDevice
        )

    def data_ptr(self):
        return self.ptr

def to_device(arr: np.ndarray):
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor

def to_host(tensor: DeviceTensor):
    return tensor.to_numpy()