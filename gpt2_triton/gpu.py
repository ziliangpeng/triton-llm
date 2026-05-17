"""
GPU Memory Allocator (ctypes + CUDA/HIP Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.

Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) backends. The backend
is detected automatically at import time.
"""

import ctypes
import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_SUCCESS = 0


def _load_runtime():
    """
    Try to load CUDA or HIP runtime library.
    Returns (lib, backend_name) or raises RuntimeError if neither is found.
    """
    # Try NVIDIA CUDA first
    for lib_name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
        try:
            lib = ctypes.CDLL(lib_name)
            _setup_argtypes(lib, prefix="cuda")
            return lib, "cuda"
        except OSError:
            continue

    # Try AMD HIP/ROCm
    for lib_name in ["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5"]:
        try:
            lib = ctypes.CDLL(lib_name)
            _setup_argtypes(lib, prefix="hip")
            return lib, "hip"
        except OSError:
            continue

    raise RuntimeError(
        "No GPU runtime found. Ensure CUDA (libcudart.so) or "
        "ROCm (libamdhip64.so) is installed."
    )


def _setup_argtypes(lib, prefix):
    """Set up argtypes for Malloc, Free, Memcpy."""
    getattr(lib, f"{prefix}Malloc").argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
    ]
    getattr(lib, f"{prefix}Malloc").restype = ctypes.c_int

    getattr(lib, f"{prefix}Free").argtypes = [ctypes.c_void_p]
    getattr(lib, f"{prefix}Free").restype = ctypes.c_int

    getattr(lib, f"{prefix}Memcpy").argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    getattr(lib, f"{prefix}Memcpy").restype = ctypes.c_int


# Load runtime at module import time
_rt, BACKEND = _load_runtime()
_malloc = getattr(_rt, f"{BACKEND}Malloc")
_free = getattr(_rt, f"{BACKEND}Free")
_memcpy = getattr(_rt, f"{BACKEND}Memcpy")


def check_error(err):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"GPU runtime error ({BACKEND}): {err}")


class DeviceTensor:
    """Wrapper for a GPU device pointer with automatic cleanup."""

    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def __del__(self):
        """Automatically free GPU memory when object is garbage collected."""
        if self.ptr and self.ptr.value is not None:
            try:
                check_error(_free(self.ptr))
            except Exception:
                pass  # Avoid errors during interpreter shutdown

    def to_numpy(self):
        """Copy from device to host."""
        host = np.empty(self.shape, dtype=self.dtype)
        check_error(
            _memcpy(
                host.ctypes.data_as(ctypes.c_void_p),
                self.ptr,
                self.nbytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        )
        return host

    def from_numpy(self, arr: np.ndarray):
        """Copy from host to device."""
        check_error(
            _memcpy(
                self.ptr,
                arr.ctypes.data_as(ctypes.c_void_p),
                self.nbytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        )

    def data_ptr(self):
        """Return the raw device pointer value."""
        return self.ptr.value


def allocate(shape, dtype=np.float32):
    """Allocate memory on GPU."""
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    ptr = ctypes.c_void_p()
    check_error(_malloc(ctypes.byref(ptr), nbytes))
    return DeviceTensor(ptr, shape, dtype, nbytes)


def to_device(arr: np.ndarray):
    """Copy numpy array to GPU."""
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor


def to_host(tensor: DeviceTensor):
    """Copy DeviceTensor back to numpy."""
    return tensor.to_numpy()