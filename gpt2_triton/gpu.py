"""
GPU Memory Allocator (ctypes + CUDA/HIP Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.

Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) backends with
environment variable override and runtime auto-detection.
"""

import ctypes
import os
import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_SUCCESS = 0


def _is_cuda_available():
    """Check if CUDA runtime is available."""
    for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _is_hip_available():
    """Check if HIP/ROCm runtime is available."""
    for name in ["libamdhip64.so", "libamdhip64.so.6"]:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _detect_backend():
    """
    Detect the GPU backend to use.

    Priority:
    1. Explicit environment variable (GPU_BACKEND=cuda or hip) - strict check
    2. Runtime detection (check /dev/kfd first for AMD)
    """
    env = os.environ.get("GPU_BACKEND", "").lower().strip()

    if env == "cuda":
        if _is_cuda_available():
            return "cuda"
        else:
            raise RuntimeError(
                "GPU_BACKEND=cuda was set, but CUDA runtime "
                "(libcudart.so) was not found on this system."
            )

    if env == "hip":
        if _is_hip_available():
            return "hip"
        else:
            raise RuntimeError(
                "GPU_BACKEND=hip was set, but HIP runtime "
                "(libamdhip64.so) was not found on this system."
            )

    # No environment variable set → auto detect
    if os.path.exists("/dev/kfd"):
        return "hip"

    if _is_cuda_available():
        return "cuda"

    if _is_hip_available():
        return "hip"

    raise RuntimeError(
        "No supported GPU runtime found. "
        "Install CUDA (libcudart.so) or ROCm (libamdhip64.so)."
    )


def _load_runtime(backend):
    """Load the appropriate runtime and set up argtypes."""
    if backend == "cuda":
        for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                lib = ctypes.CDLL(name)
                _setup_argtypes(lib, "cuda")
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load CUDA runtime")

    elif backend == "hip":
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                lib = ctypes.CDLL(name)
                _setup_argtypes(lib, "hip")
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load HIP runtime")

    raise RuntimeError(f"Unknown backend: {backend}")


def _setup_argtypes(lib, prefix):
    """Set ctypes argtypes for type safety and 64-bit compatibility."""
    malloc_fn = getattr(lib, f"{prefix}Malloc")
    free_fn = getattr(lib, f"{prefix}Free")
    memcpy_fn = getattr(lib, f"{prefix}Memcpy")

    malloc_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    malloc_fn.restype = ctypes.c_int

    free_fn.argtypes = [ctypes.c_void_p]
    free_fn.restype = ctypes.c_int

    memcpy_fn.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    memcpy_fn.restype = ctypes.c_int

    return malloc_fn, free_fn, memcpy_fn


# Lazy initialization state
_initialized = False
_backend = None
_rt = None
_malloc = None
_free = None
_memcpy = None


def _initialize():
    """Lazily initialize the GPU runtime on first use."""
    global _initialized, _backend, _rt, _malloc, _free, _memcpy

    if _initialized:
        return

    _backend = _detect_backend()
    _rt = _load_runtime(_backend)

    if _backend == "cuda":
        _malloc, _free, _memcpy = _setup_argtypes(_rt, "cuda")
    else:
        _malloc, _free, _memcpy = _setup_argtypes(_rt, "hip")

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
        if self.ptr and self.ptr.value is not None:
            try:
                if _free is not None:
                    _free(self.ptr)
            except Exception:
                pass

    def to_numpy(self):
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