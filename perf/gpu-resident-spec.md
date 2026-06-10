# GPU-Resident Inference — EC32 Spec

## Problem

Every kernel call currently does:
```
to_device(h) + to_device(weight) → kernel launch → synchronize → to_host(out)
```
For SmolLM2-135M (30 layers × ~15 kernel calls/layer + overhead), that's ~420 CPU↔GPU round-trips per decode step, each doing a full device-wide sync. This dominates TPOT (~250ms) despite the model being only 135M params.

## Solution: Three Phases (Phase 1 = EC32)

**Phase 1 — GPU-resident weights + no per-kernel sync:**
- Pre-allocate all weights as `DeviceTensor` at `__init__` time (never leave GPU)
- Add device-pointer-only kernel wrappers (skip `to_device`/`to_host`/`synchronize`)
- Keep hidden state on GPU across the entire forward pass
- Only `synchronize()` once at the very end, before `to_host(logits)`
- Expected TPOT: ~250ms → ~15-30ms

**Phase 2 — GPU-resident KV cache (separate PR):**
- KV cache lives on GPU, written/read directly via device pointers
- Eliminates numpy reshape/copy on cache slices (the `[:, :total_after, :].reshape(-1, d_k)` CPU copy)
- Requires GQA kernel to accept custom head stride

**Phase 3 — Fused decode loop (separate PR):**
- Fuse all per-token kernels into a single launch where possible
- No synchronize at all until the final token

---

## EC32: Phase 1 Detailed Design

### 1. GPU Weight Storage

In `SmolLM2ForCausalLM.__init__()`, store weights as `gpu.DeviceTensor`:

```python
self.q_proj_w_dev: list[DeviceTensor] = []
# ...
for i in range(n_layer):
    w = np.require(weights[...].T.copy(), dtype=np.float32, requirements=['C_CONTIGUOUS'])
    self.q_proj_w_dev.append(gpu.to_device(w))
```

Keep CPU-side copies only for weights that need CPU operations (token embedding wte — used for numpy indexing). GPU copies go on `_w_dev` lists.

### 2. Kernel Wrapper Changes

Add a `*_device()` variant to each kernel module. These skip to_device/to_host/synchronize:

**`triton_llm/kernels/gemm.py`** — Add `gemm_device`:
```python
def gemm_device(
    h_dev: gpu.DeviceTensor,          # (M, K) on GPU
    w_dev: gpu.DeviceTensor,          # (K, N) on GPU
    out_dev: gpu.DeviceTensor | None = None,  # (M, N) on GPU, or None to auto-allocate
) -> gpu.DeviceTensor:
    """GPU-resident GEMM. No sync, no host copies."""
    M, K = h_dev.shape
    K2, N = w_dev.shape
    assert K == K2
    if out_dev is None:
        out_dev = gpu.allocate((M, N), np.float32)
    _gemm_kernel[M//BLOCK_M, N//BLOCK_N](
        h_dev.data_ptr(), w_dev.data_ptr(), out_dev.data_ptr(),
        M, N, K,
        N, N, N,  # strides (contiguous)
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        GROUP_M=8,
    )
    return out_dev  # no sync, no to_host
```

**`triton_llm/kernels/rms_norm.py`** — Add `rms_norm_device`:
```python
def rms_norm_device(
    x_dev: gpu.DeviceTensor,    # (M, N) on GPU
    w_dev: gpu.DeviceTensor,    # (N,) weight on GPU
    out_dev: gpu.DeviceTensor,  # (M, N) on GPU
    eps: float = 1e-5,
) -> gpu.DeviceTensor:
    """GPU-resident RMSNorm. No sync, no host copies."""
    M, N = x_dev.shape
    BLOCK_SIZE = max(_next_pow2(N), 16)
    _rms_norm_kernel[(M,)](
        x_dev.data_ptr(), out_dev.data_ptr(), w_dev.data_ptr(),
        M, N, N, N, eps, BLOCK_SIZE,
    )
    return out_dev
```

**`triton_llm/kernels/rope.py`** — Add `apply_rope_device`:
Same pattern: takes device pointers for x, cos, sin, returns out on device.

**`triton_llm/kernels/swiglu.py`** — Add `swiglu_device`:
```python
def swiglu_device(gate_dev, up_dev, out_dev) -> DeviceTensor:
    _swiglu_kernel[grid](gate_dev.data_ptr(), up_dev.data_ptr(), out_dev.data_ptr(), ...)
    return out_dev
```

**`triton_llm/kernels/add.py`** — Add `add_device`:
```python
def add_device(x_dev, y_dev, out_dev) -> DeviceTensor:
    _add_kernel[grid](x_dev.data_ptr(), y_dev.data_ptr(), out_dev.data_ptr(), ...)
    return out_dev
```

**`triton_llm/kernels/attention_gqa.py`** — Add `attention_gqa_device`:
Key insight: K/V can be passed as device pointers directly. For cached decode, `k_view` and `v_view` can be pre-allocated device tensors or sub-slices.

### 3. Model Forward Pass Changes

The core change: `_forward_cached` keeps `h` as a `DeviceTensor` across all layers instead of converting to/from numpy.

```python
def _forward_cached_gpu(self, token_ids: np.ndarray) -> np.ndarray:
    # Embed on CPU (numpy indexing), copy to GPU once
    hidden = self._embed(token_ids)  # (1, seq, n_embd) on CPU
    h_dev = gpu.to_device(hidden.reshape(-1, n_embd))  # (seq, n_embd) on GPU

    for i in range(n_layer):
        # RMSNorm on GPU
        out_dev = gpu.allocate((seq, n_embd), np.float32)
        rms_norm_device(h_dev, self.ln_1_w_dev[i], out_dev, eps)
        
        # QKV projections on GPU
        q_dev = gemm_device(out_dev, self.q_proj_w_dev[i])
        k_dev = gemm_device(out_dev, self.k_proj_w_dev[i])
        v_dev = gemm_device(out_dev, self.v_proj_w_dev[i])
        
        # RoPE on GPU
        cos_dev = self.cos_dev  # already on GPU
        sin_dev = self.sin_dev
        apply_rope_device(q_dev, cos_dev, sin_dev, seq_len=seq, position_offset=prev_seq)
        apply_rope_device(k_dev, cos_dev, sin_dev, seq_len=seq, position_offset=prev_seq)
        
        # Attention (GQA takes device pointers directly — already works on GPU)
        attn_out_dev = attention_gqa_device(q_dev, k_dev, v_dev, n_head, n_kv_head, causal=...)
        
        # Output projection + residual on GPU
        o_dev = gemm_device(..., self.o_proj_w_dev[i])
        add_device(o_dev, h_dev, ...)  # residual

        # MLP on GPU
        # rms_norm_device → gemm_device(gate) + gemm_device(up) → swiglu_device → gemm_device(down)
        # add_device(residual)

    # Final norm + LM head on GPU
    final_dev = rms_norm_device(h_dev, self.ln_f_w_dev, ...)
    logits_dev = gemm_device(final_dev, self.lm_head_w_dev)
    
    # Single sync + to_host at the very end
    gpu.synchronize()
    return gpu.to_host(logits_dev).reshape(1, seq, vocab_size)
```

### 4. Memory Budget Check

SmolLM2-135M weights (FP32):
- Embedding: 49152 × 576 = 108 MB
- Per layer (30 layers):
  - Q: 576 × 576 = 1.26 MB × 30 = 37.8 MB
  - K: 576 × 192 = 0.42 MB × 30 = 12.6 MB
  - V: 576 × 192 = 0.42 MB × 30 = 12.6 MB
  - O: 576 × 576 = 1.26 MB × 30 = 37.8 MB
  - gate: 576 × 1536 = 3.39 MB × 30 = 101.7 MB
  - up: 576 × 1536 = 3.39 MB × 30 = 101.7 MB
  - down: 1536 × 576 = 3.39 MB × 30 = 101.7 MB
  - ln_1: 576 × 1 = 0.07 MB
  - ln_2: 576 × 1 = 0.07 MB
- Total: ~525 MB for weights + RoPE cos/sin (~2 MB) + scratch buffers
H100 has 80 GB. ✅

### 5. Files to Modify

| File | Change |
|------|--------|
| `triton_llm/kernels/gemm.py` | Add `gemm_device()` |
| `triton_llm/kernels/rms_norm.py` | Add `rms_norm_device()` |
| `triton_llm/kernels/rope.py` | Add `apply_rope_device()` + precompute cos/sin on GPU |
| `triton_llm/kernels/swiglu.py` | Add `swiglu_device()` |
| `triton_llm/kernels/add.py` | Add `add_device()` |
| `triton_llm/kernels/attention_gqa.py` | Add `attention_gqa_device()` |
| `smollm2_triton/model.py` | GPU-resident forward pass |
| `tests/test_smollm2_model.py` | GPU-resident correctness test (add `test_gpu_resident_full_trip`) |
| `gpu.py` | (Maybe) Add `zeros()` helper for DeviceTensor |

### 6. Backward Compatibility

Existing CPU-based functions (`gemm()`, `rms_norm()`, etc.) remain unchanged. The new `*_device()` functions are additive. All existing tests continue to pass.

### 7. Correctness Verification

- `test_gpu_resident_full_trip`: generate 5 tokens with GPU-resident path, compare output token-by-token vs CPU-based `generate()`
- Cosine similarity > 0.999 on intermediate hidden states
- All existing 109 tests still pass

### 8. Naming Convention

New functions in each kernel module:
```python
def gemm_device(h_dev, w_dev, out_dev=None) -> DeviceTensor: ...
def rms_norm_device(x_dev, w_dev, out_dev, eps=1e-5) -> DeviceTensor: ...
def apply_rope_device(x_dev, cos_dev, sin_dev, seq_len, position_offset=0) -> DeviceTensor: ...
def swiglu_device(gate_dev, up_dev, out_dev=None) -> DeviceTensor: ...
def add_device(x_dev, y_dev, out_dev=None) -> DeviceTensor: ...
def attention_gqa_device(q_dev, k_dev, v_dev, n_head, n_kv_head, causal=True) -> DeviceTensor: ...
```

Each returns a `DeviceTensor` (not numpy), performs no `synchronize()`, performs no `to_host()`.
