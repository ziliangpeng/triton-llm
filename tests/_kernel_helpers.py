"""Helpers to test GPU-only kernels with numpy arrays.

Each function wraps a DeviceTensor-based kernel with to_device/to_host
so that tests can pass numpy arrays and get numpy arrays back, just
like the old CPU-wrappers did.

This is a testing-only layer — it moves data to GPU, calls the kernel,
synchronizes, and moves the result back.
"""

import numpy as np
from triton_llm import gpu
from triton_llm.kernels.gemm import gemm as _gemm
from triton_llm.kernels.rms_norm import rms_norm as _rms_norm
from triton_llm.kernels.rope import apply_rope as _apply_rope
from triton_llm.kernels.swiglu import swiglu as _swiglu
from triton_llm.kernels.add import add as _add
from triton_llm.kernels.attention_gqa import attention_gqa as _attention_gqa


def gemm_cpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Test helper: call GPU gemm with numpy arrays."""
    a_dev = gpu.to_device(np.require(a, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    b_dev = gpu.to_device(np.require(b, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    out_dev = _gemm(a_dev, b_dev)
    gpu.synchronize()
    return gpu.to_host(out_dev)


def rms_norm_cpu(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Test helper: call GPU rms_norm with numpy arrays.

    Flattens leading dimensions for inputs with ndim > 2, then reshapes
    the result back to the original shape.
    """
    if x.ndim < 2:
        raise ValueError(f"RMSNorm expects at least a 2D input, got {x.ndim}D")
    orig_shape = x.shape
    needs_reshape = x.ndim > 2
    if needs_reshape:
        x = x.reshape(-1, orig_shape[-1])
    x_dev = gpu.to_device(np.require(x, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    w_dev = gpu.to_device(np.require(weight, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    out_dev = gpu.allocate(x.shape, np.float32)
    _rms_norm(x_dev, w_dev, out_dev, eps=eps)
    gpu.synchronize()
    out = gpu.to_host(out_dev)
    if needs_reshape:
        out = out.reshape(orig_shape)
    return out


def apply_rope_cpu(
    x: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    seq_len: int,
    position_offset: int = 0,
) -> np.ndarray:
    """Test helper: call GPU apply_rope with numpy arrays.

    Flattens leading dimensions for inputs with ndim > 2, then reshapes
    the result back to the original shape.

    Note: cos/sin must be numpy arrays from precompute_cos_sin().
    They are moved to GPU inside this helper.
    """
    orig_shape = x.shape
    needs_reshape = x.ndim > 2
    if needs_reshape:
        x = x.reshape(-1, orig_shape[-1])
    x_dev = gpu.to_device(np.require(x, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    cos_dev = gpu.to_device(np.require(cos, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    sin_dev = gpu.to_device(np.require(sin, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    _apply_rope(x_dev, cos_dev, sin_dev, seq_len=seq_len, position_offset=position_offset)
    gpu.synchronize()
    out = gpu.to_host(x_dev)
    if needs_reshape:
        out = out.reshape(orig_shape)
    return out


def swiglu_cpu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Test helper: call GPU swiglu with numpy arrays."""
    gate_dev = gpu.to_device(np.require(gate, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    up_dev = gpu.to_device(np.require(up, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    out_dev = _swiglu(gate_dev, up_dev)
    gpu.synchronize()
    return gpu.to_host(out_dev)


def add_cpu(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Test helper: call GPU add with numpy arrays."""
    x_dev = gpu.to_device(np.require(x, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    y_dev = gpu.to_device(np.require(y, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    out_dev = _add(x_dev, y_dev)
    gpu.synchronize()
    return gpu.to_host(out_dev)


def attention_gqa_cpu(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    n_head: int,
    n_kv_head: int,
    causal: bool = True,
) -> np.ndarray:
    """Test helper: call GPU attention_gqa with numpy arrays."""
    q_dev = gpu.to_device(np.require(q, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    k_dev = gpu.to_device(np.require(k, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    v_dev = gpu.to_device(np.require(v, dtype=np.float32, requirements=["C_CONTIGUOUS"]))
    o_dev = _attention_gqa(q_dev, k_dev, v_dev, n_head, n_kv_head, causal=causal)
    gpu.synchronize()
    return gpu.to_host(o_dev)
