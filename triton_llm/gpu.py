"""
GPU Memory Allocator (ctypes + CUDA/HIP Runtime)

Provides a minimal PyTorch-free interface to allocate and transfer
memory between host and device.

Supports both NVIDIA (CUDA) and AMD (HIP/ROCm) backends with
environment variable override and runtime auto-detection.
"""

import ctypes
import os
import getpass
import numpy as np

CUDA_MEMCPY_HOST_TO_DEVICE = 1
CUDA_MEMCPY_DEVICE_TO_HOST = 2
CUDA_SUCCESS = 0

# Common library names for each backend
CUDA_LIBRARIES = ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]
HIP_LIBRARIES = ["libamdhip64.so", "libamdhip64.so.6"]


def _is_cuda_available():
    """Check if any CUDA runtime library is present."""
    for name in CUDA_LIBRARIES:
        try:
            ctypes.CDLL(name)
            return True
        except OSError:
            continue
    return False


def _is_hip_available():
    """Check if any HIP/ROCm runtime library is present."""
    for name in HIP_LIBRARIES:
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
    1. Explicit environment variable (GPU_BACKEND=cuda or hip) - strict validation
    2. Runtime auto-detection
    """
    env = os.environ.get("GPU_BACKEND", "").lower().strip()

    if env == "cuda":
        if _is_cuda_available():
            return "cuda"
        else:
            raise RuntimeError(
                "GPU_BACKEND=cuda was set, but no CUDA runtime library was found."
            )

    if env == "hip":
        if _is_hip_available():
            return "hip"
        else:
            raise RuntimeError(
                "GPU_BACKEND=hip was set, but no HIP runtime library was found."
            )

    # Auto detection when no env var is set
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
    """Load the appropriate runtime library and set up argtypes."""
    if backend == "cuda":
        for name in CUDA_LIBRARIES:
            try:
                lib = ctypes.CDLL(name)
                _setup_argtypes(lib, "cuda")
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load any CUDA runtime library")

    elif backend == "hip":
        for name in HIP_LIBRARIES:
            try:
                lib = ctypes.CDLL(name)
                _setup_argtypes(lib, "hip")
                return lib
            except OSError:
                continue
        raise RuntimeError("Failed to load any HIP runtime library")

    raise RuntimeError(f"Unknown backend: {backend}")


def _setup_argtypes(lib, prefix):
    """Configure argtypes and restype for malloc/free/memcpy."""
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
    """Lazily initialize the GPU runtime on first use.

    Also sets ``TRITON_CACHE_DIR`` to a local (non-NFS) path so that
    Triton JIT compilation and cache lookups don't go through the
    network filesystem — on clusters where ``$HOME`` is NFS-mounted,
    that adds ~2 s of latency to the first forward pass.
    """
    global _initialized, _backend, _rt, _malloc, _free, _memcpy

    if _initialized:
        return

    # Force Triton's JIT cache onto a local (non-NFS) filesystem.
    # The default ``~/.triton/cache/`` lives on the home directory,
    # which is NFS-mounted on many GPU clusters (gcp5, amd2, etc.).
    # Every cache lookup over NFS adds ~80–100 µs, and the first
    # access can stall for seconds.
    if "TRITON_CACHE_DIR" not in os.environ:
        try:
            username = getpass.getuser()
            os.environ["TRITON_CACHE_DIR"] = f"/tmp/triton_cache_{username}"
        except Exception:
            os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"

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

    def __init__(self, ptr, shape, dtype, nbytes, defer_free=True):
        self.ptr = ptr
        self.shape = shape
        self.dtype = dtype
        self.nbytes = nbytes
        self._defer_free = defer_free

    def __del__(self):
        if self._defer_free and self.ptr and self.ptr.value is not None:
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


def synchronize():
    """Synchronize the current device (CUDA or HIP).

    Ensures all previous GPU operations are complete before proceeding.
    Critical after kernel launches before host-side reads.
    """
    _initialize()
    sync_fn = getattr(_rt, f"{_backend}DeviceSynchronize")
    sync_fn.argtypes = []
    sync_fn.restype = ctypes.c_int
    err = sync_fn()
    check_error(err)


def view(tensor: DeviceTensor, new_shape: tuple) -> DeviceTensor:
    """Return a zero-copy view of ``tensor`` with a new shape.

    Same device pointer, same memory, different shape tuple.  The new
    shape must have the same total number of elements as the original.
    The returned tensor does NOT own the GPU memory — freeing is
    deferred to the original tensor.

    Parameters
    ----------
    tensor : DeviceTensor
        Source tensor whose GPU memory is reused.
    new_shape : tuple of int
        New shape (same total element count).

    Returns
    -------
    DeviceTensor
        A view with the same pointer and ``nbytes`` but shape ``new_shape``.
        ``defer_free=False`` so the original tensor is responsible for freeing.
    """
    n_elems = int(np.prod(tensor.shape))
    n_new = int(np.prod(new_shape))
    if n_elems != n_new:
        raise ValueError(
            f"Cannot view {tensor.shape} ({n_elems} elements) as "
            f"{new_shape} ({n_new} elements) — element count mismatch"
        )
    return DeviceTensor(tensor.ptr, new_shape, tensor.dtype, tensor.nbytes, defer_free=False)