"""
GPU Memory Allocator (ctypes + CUDA/HIP Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.

Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) backends.
"""

import ctypes
import os
import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_SUCCESS = 0


def _detect_backend():
    """
    Detect whether we are on NVIDIA (CUDA) or AMD (HIP/ROCm).
    Priority: explicit env var > /dev/kfd (AMD) > library presence.
    """
    env = os.environ.get("GPU_BACKEND", "").lower()
    if env in ("cuda", "hip"):
        return env

    # AMD ROCm systems expose /dev/kfd
    if os.path.exists("/dev/kfd"):
        return "hip"

    # Try HIP libraries first on AMD-like environments
    for lib_name in ["libamdhip64.so", "libamdhip64.so.6"]:
        try:
            ctypes.CDLL(lib_name)
            return "hip"
        except OSError:
            continue

    # Fallback to CUDA
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


def _load_runtime(backend):
    if backend == "cuda":
        for name in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                lib = ctypes.CDLL(name)
                _setup_cuda_argtypes(lib)
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load CUDA runtime")

    elif backend == "hip":
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                lib = ctypes.CDLL(name)
                _setup_hip_argtypes(lib)
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load HIP runtime")

    raise RuntimeError(f"Unknown backend: {backend}")


def _setup_cuda_argtypes(lib):
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.cudaMemcpy.restype = ctypes.c_int


def _setup_hip_argtypes(lib):
    lib.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.hipMalloc.restype = ctypes.c_int
    lib.hipFree.argtypes = [ctypes.c_void_p]
    lib.hipFree.restype = ctypes.c_int
    lib.hipMemcpy.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    ]
    lib.hipMemcpy.restype = ctypes.c_int


BACKEND = _detect_backend()
_rt = _load_runtime(BACKEND)

if BACKEND == "cuda":
    _malloc = _rt.cudaMalloc
    _free = _rt.cudaFree
    _memcpy = _rt.cudaMemcpy
elif BACKEND == "hip":
    _malloc = _rt.hipMalloc
    _free = _rt.hipFree
    _memcpy = _rt.hipMemcpy


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
        if self.ptr and self.ptr.value is not None:
            try:
                check_error(_free(self.ptr))
            except Exception:
                pass

    def to_numpy(self):
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
    nbytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
    ptr = ctypes.c_void_p()
    check_error(_malloc(ctypes.byref(ptr), nbytes))
    return DeviceTensor(ptr, shape, dtype, nbytes)


def to_device(arr: np.ndarray):
    tensor = allocate(arr.shape, arr.dtype)
    tensor.from_numpy(arr)
    return tensor


def to_host(tensor: DeviceTensor):
    return tensor.to_numpy()