"""
GPU Memory Allocator (ctypes + CUDA/HIP Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.

Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) backends with lazy
initialization to avoid import-time failures on non-GPU systems.
"""

import ctypes
import os
import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_SUCCESS = 0

# Module-level state for lazy initialization
_initialized = False
_backend = None
_rt = None
_malloc = None
_free = None
_memcpy = None


def _detect_backend():
    """Detect GPU backend. Priority: env var > /dev/kfd > library presence."""
    env = os.environ.get("GPU_BACKEND", "").lower()
    if env in ("cuda", "hip"):
        return env

    if os.path.exists("/dev/kfd"):
        return "hip"

    for lib_name in ["libamdhip64.so", "libamdhip64.so.6"]:
        try:
            ctypes.CDLL(lib_name)
            return "hip"
        except OSError:
            continue

    for lib_name in ["libcudart.so", "libcudart.so.12"]:
        try:
            ctypes.CDLL(lib_name)
            return "cuda"
        except OSError:
            continue

    raise RuntimeError(
        "No supported GPU runtime found. "
        "Install CUDA (libcudart.so) or ROCm (libamdhip64.so)."
    )


def _setup_argtypes(lib, prefix):
    """Set argtypes for malloc/free/memcpy to ensure 64-bit safety."""
    malloc_func = getattr(lib, f"{prefix}Malloc")
    free_func = getattr(lib, f"{prefix}Free")
    memcpy_func = getattr(lib, f"{prefix}Memcpy")

    malloc_func.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    malloc_func.restype = ctypes.c_int

    free_func.argtypes = [ctypes.c_void_p]
    free_func.restype = ctypes.c_int

    memcpy_func.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    memcpy_func.restype = ctypes.c_int

    return malloc_func, free_func, memcpy_func


def _initialize():
    """Lazily initialize the GPU runtime. Called on first use."""
    global _initialized, _backend, _rt, _malloc, _free, _memcpy

    if _initialized:
        return

    _backend = _detect_backend()

    if _backend == "cuda":
        for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                _rt = ctypes.CDLL(name)
                _malloc, _free, _memcpy = _setup_argtypes(_rt, "cuda")
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Failed to load CUDA runtime")

    elif _backend == "hip":
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                _rt = ctypes.CDLL(name)
                _malloc, _free, _memcpy = _setup_argtypes(_rt, "hip")
                break
            except OSError:
                continue
        else:
            raise RuntimeError("Failed to load HIP runtime")

    _initialized = True


def check_error(err):
    if err != CUDA_SUCCESS:
        raise RuntimeError(f"GPU runtime error ({_backend}): {err}")


class DeviceTensor:
    """Wrapper for a GPU device pointer with automatic cleanup."""

    def __init__(self, ptr, shape, dtype, nbytes):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes

    def __del__(self):
        """Free GPU memory. Use a local reference to avoid global issues."""
        if self.ptr and self.ptr.value is not None:
            try:
                # Call free directly if available
                if _free is not None:
                    _free(self.ptr)
            except Exception:
                pass

    def to_numpy(self):
        """Copy from device to host."""
        _initialize()
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
        _initialize()
        check_error(
            _memcpy(
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
    _initialize()
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    ptr = ctypes.c_void_p()
    check_error(_malloc(ctypes.byref(ptr), nbytes))
    return DeviceTensor(ptr, shape, dtype, nbytes)


def to_device(arr: np.ndarray):
    """Copy numpy array to GPU."""
    _initialize()
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor


def to_host(tensor: DeviceTensor):
    """Copy DeviceTensor back to numpy."""
    return tensor.to_numpy()