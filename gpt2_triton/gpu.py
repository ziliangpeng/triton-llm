"""
GPU Memory Allocator (ctypes + CUDA Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.
"""

import ctypes
import numpy as np

# Load CUDA runtime
cudart = ctypes.CDLL("libcudart.so")

# Define argtypes for safety and 64-bit compatibility
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int

cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int

cudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]
cudart.cudaMemcpy.restype = ctypes.c_int

CUDA_SUCCESS = 0
CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2


def check_cuda(err):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error: {err}")


class DeviceTensor:
    """Wrapper for a CUDA device pointer with automatic cleanup."""

    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def __del__(self):
        """Automatically free GPU memory when object is garbage collected."""
        if self.ptr and self.ptr.value is not None:
            try:
                check_cuda(cudart.cudaFree(self.ptr))
            except Exception:
                pass  # Avoid errors during interpreter shutdown

    def to_numpy(self):
        """Copy from device to host."""
        host = np.empty(self.shape, dtype=self.dtype)
        check_cuda(
            cudart.cudaMemcpy(
                host.ctypes.data_as(ctypes.c_void_p),
                self.ptr,
                self.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        )
        return host

    def from_numpy(self, arr: np.ndarray):
        """Copy from host to device."""
        check_cuda(
            cudart.cudaMemcpy(
                self.ptr,
                arr.ctypes.data_as(ctypes.c_void_p),
                self.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        )

    def data_ptr(self):
        return self.ptr.value


def allocate(shape, dtype=np.float32):
    """Allocate memory on GPU."""
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    ptr = ctypes.c_void_p()
    check_cuda(cudart.cudaMalloc(ctypes.byref(ptr), nbytes))
    return DeviceTensor(ptr, shape, dtype, nbytes)


def to_device(arr: np.ndarray):
    """Copy numpy array to GPU."""
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor


def to_host(tensor: DeviceTensor):
    """Copy DeviceTensor back to numpy."""
    return tensor.to_numpy()