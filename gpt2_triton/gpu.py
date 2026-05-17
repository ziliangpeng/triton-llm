"""
GPU Memory Allocator (ctypes + CUDA Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.
"""

import ctypes
import numpy as np

cudart = ctypes.CDLL("libcudart.so")
CUDA_SUCCESS = 0


def check_cuda(err):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"CUDA error: {err}")


class DeviceTensor:
    """Wrapper for a CUDA device pointer."""

    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def to_numpy(self):
        """Copy from device to host."""
        host = np.empty(self.shape, dtype=self.dtype)
        check_cuda(
            cudart.cudaMemcpy(
                host.ctypes.data_as(ctypes.c_void_p),
                self.ptr,
                self.nbytes,
                ctypes.c_int(2),  # DeviceToHost
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
                ctypes.c_int(1),  # HostToDevice
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