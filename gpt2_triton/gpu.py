"""
GPU memory management using only ctypes + CUDA runtime (no PyTorch, minimal Triton dependency).
"""

import numpy as np
import ctypes

# Load CUDA runtime
try:
    cudart = ctypes.CDLL('libcudart.so')
except OSError:
    cudart = ctypes.CDLL('libcudart.so.12')

cudaSuccess = 0

def check_cuda(error):
    if error != cudaSuccess:
        raise RuntimeError(f"CUDA error: {error}")

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
            ctypes.c_int(2)  # cudaMemcpyDeviceToHost
        ))
        return host

    def from_numpy(self, arr: np.ndarray):
        check_cuda(cudart.cudaMemcpy(
            self.ptr,
            arr.ctypes.data_as(ctypes.c_void_p),
            self.nbytes,
            ctypes.c_int(1)  # cudaMemcpyHostToDevice
        ))

    def data_ptr(self):
        return self.ptr.value

def to_device(arr: np.ndarray):
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor

def to_host(tensor: DeviceTensor):
    return tensor.to_numpy()