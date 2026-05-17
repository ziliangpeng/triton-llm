"""
Minimal GPU memory management using only ctypes + CUDA runtime.
This version is known to work for allocate + memcpy.
"""

import numpy as np
import ctypes

cudart = ctypes.CDLL('libcudart.so')

cudaSuccess = 0

def check_cuda(err):
    if err != cudaSuccess:
        raise RuntimeError(f"CUDA error {err}")

def allocate(shape, dtype=np.float32):
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    ptr = ctypes.c_void_p()
    check_cuda(cudart.cudaMalloc(ctypes.byref(ptr), nbytes))
    return DeviceTensor(ptr, shape, dtype, nbytes)

class DeviceTensor:
    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def to_numpy(self):
        host = np.empty(self.shape, dtype=self.dtype)
        check_cuda(cudart.cudaMemcpy(
            host.ctypes.data_as(ctypes.c_void_p),
            self.ptr,
            self.nbytes,
            ctypes.c_int(2)  # DeviceToHost
        ))
        return host

    def from_numpy(self, arr: np.ndarray):
        check_cuda(cudart.cudaMemcpy(
            self.ptr,
            arr.ctypes.data_as(ctypes.c_void_p),
            self.nbytes,
            ctypes.c_int(1)  # HostToDevice
        ))

    def data_ptr(self):
        return self.ptr.value

def to_device(arr: np.ndarray):
    t = allocate(arr.shape, arr.dtype)
    t.from_numpy(arr)
    return t

def to_host(t: DeviceTensor):
    return t.to_numpy()