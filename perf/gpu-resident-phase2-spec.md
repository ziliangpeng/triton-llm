# GPU-Resident Inference — Phase 2: GPU Transpose + GPU KV Cache

## Problem

Phase 1 (EC32) showed: TPOT 181ms → 73ms (2.5x), but TTFT 224ms → 1330ms (6x worse).
The bottleneck is `to_host`/`to_device` for QKV reshape+transpose and KV cache interaction:
- **4-6 round trips per layer** for QKV head-major reshape/transpose (both directions)
- **2-4 round trips per layer** for KV cache read/write
- 30 layers × ~8 trips/layer = 240 DMA transfers per forward call

Each DMA transfer is synchronous (kernel → host, or host → kernel), which dominates the prefill time.

## Solution

### 1. GPU Transpose Kernel (`triton_llm/kernels/transpose_2d.py`)

Two simple copy kernels that never leave the GPU:

```
to_head_major:  (seq, n_heads*d_k) → (n_heads*seq, d_k)
                Reading: in[seq_pos * n_heads * d_k + head * d_k + dim]
                Writing: out[(head * seq + seq_pos) * d_k + dim]
                Grid: (n_heads * seq,) programs, each copies 1 row of d_k elements

to_seq_major:   (n_heads*seq, d_k) → (seq, n_heads*d_k)
                Reading: in[(head * seq + seq_pos) * d_k + dim]
                Writing: out[seq_pos * n_heads * d_k + head * d_k + dim]
                Grid: (n_heads * seq,) programs, each copies 1 row of d_k elements
```

Implementation:
```python
@triton.jit
def _to_head_major_kernel(I, O, seq, n_heads, HEAD_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    head = pid // seq
    seq_pos = pid % seq
    # Input: (seq, n_heads * d_k) — offset = seq_pos * n_heads * d_k + head * d_k
    in_off = seq_pos * n_heads * HEAD_SIZE + head * HEAD_SIZE
    # Output: (n_heads * seq, d_k) — offset = (head * seq + seq_pos) * d_k
    out_off = (head * seq + seq_pos) * HEAD_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(I + in_off + offs, mask=mask)
    tl.store(O + out_off + offs, x, mask=mask)

def to_head_major(x_dev, n_head, seq, d_k) -> DeviceTensor:
    """Transpose (seq, n_head*d_k) → (n_head*seq, d_k) on GPU. No sync."""
    out = gpu.allocate((n_head * seq, d_k), np.float32)
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_head * seq,)
    _to_head_major_kernel[grid](x_dev.data_ptr(), out.data_ptr(), seq, n_head, d_k, BLOCK_SIZE)
    return out
```

Similarly for `to_seq_major` (reverse direction).

### 2. GPU KV Cache

Replace numpy-based cache with DeviceTensor-based cache:

```python
# In _init_cache_gpu:
self.kv_cache_dev[i] = {
    "k": gpu.allocate((n_kv_head, max_seq, d_k), np.float32),  # DeviceTensor!
    "v": gpu.allocate((n_kv_head, max_seq, d_k), np.float32),
}
```

### 3. Cache Copy Kernels

Two copy operations between head-major flat and cache layout:

**Write to cache** (after RoPE, k_dev is `(n_kv_head*seq, d_k)` head-major flat):
```
Copy: k_dev[h*seq + p, dim] → cache["k"][h, prev_seq + p, dim]
Grid: (n_kv_head * seq,) — each program copies 1 row of d_k elements
```

**Read from cache** (for decode attention, need `(n_kv_head*total_after, d_k)` head-major flat):
```
Copy: cache["k"][h, p, dim] → k_view_dev[h*total_after + p, dim]
Grid: (n_kv_head * total_after,) — each program copies 1 row of d_k elements
```

Implementation in `transpose_2d.py` alongside the transpose functions:
```python
@triton.jit
def _cache_to_flat_kernel(CACHE, OUT, n_kv_head, total_seq, max_seq, HEAD_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // total_seq
    p = pid % total_seq
    # Cache layout: (n_kv_head, max_seq, d_k) — offset = h * max_seq * d_k + p * d_k
    cache_off = h * max_seq * HEAD_SIZE + p * HEAD_SIZE
    # Flat layout: (n_kv_head * total_seq, d_k) — offset = (h * total_seq + p) * HEAD_SIZE
    out_off = (h * total_seq + p) * HEAD_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(CACHE + cache_off + offs, mask=mask)
    tl.store(OUT + out_off + offs, x, mask=mask)

def cache_to_flat(cache_dev, n_kv_head, total_seq, d_k, max_seq):
    """Copy cache slice (n_kv_head, total_seq, d_k) → (n_kv_head * total_seq, d_k)."""
    out = gpu.allocate((n_kv_head * total_seq, d_k), np.float32)
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_kv_head * total_seq,)
    _cache_to_flat_kernel[grid](cache_dev.data_ptr(), out.data_ptr(), n_kv_head, total_seq, max_seq, d_k, BLOCK_SIZE)
    return out

@triton.jit
def _flat_to_cache_kernel(FLAT, CACHE, n_kv_head, seq, max_seq, pos_offset, HEAD_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    h = pid // seq
    p = pid % seq
    # Flat layout: (n_kv_head * seq, d_k) — offset = (h * seq + p) * HEAD_SIZE
    flat_off = (h * seq + p) * HEAD_SIZE
    # Cache layout: (n_kv_head, max_seq, d_k) — offset = h * max_seq * d_k + (pos_offset + p) * d_k
    cache_off = h * max_seq * HEAD_SIZE + (pos_offset + p) * HEAD_SIZE
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < HEAD_SIZE
    x = tl.load(FLAT + flat_off + offs, mask=mask)
    tl.store(CACHE + cache_off + offs, x, mask=mask)

def flat_to_cache(flat_dev, cache_dev, n_kv_head, seq, d_k, max_seq, pos_offset=0):
    """Copy head-major flat (n_kv_head*seq, d_k) → cache slice (n_kv_head, seq, d_k) at pos_offset."""
    BLOCK_SIZE = triton.next_power_of_2(d_k)
    grid = (n_kv_head * seq,)
    _flat_to_cache_kernel[grid](flat_dev.data_ptr(), cache_dev.data_ptr(), n_kv_head, seq, max_seq, pos_offset, d_k, BLOCK_SIZE)
```

### 4. Model Rewrite

`_forward_cached_gpu` becomes **fully GPU-resident**:

```python
def _forward_cached_gpu(self, token_ids):
    # ... same setup ...
    
    # Token embedding on CPU, then copy to GPU once
    hidden = self._embed(token_ids)
    h_dev = gpu.to_device(hidden.reshape(-1, n_embd).copy())
    
    for i in range(n_layer):
        cache = self.kv_cache_dev[i]
        residual_dev = h_dev
        
        # RMSNorm on GPU (no change)
        ln_out_dev = gpu.allocate(...)
        rms_norm_device(h_dev, self.ln_1_w_dev[i], ln_out_dev, eps)
        
        # QKV projections on GPU (no change)
        q_dev = gemm_device(ln_out_dev, self.q_proj_w_dev[i])  # (seq, n_head*d_k)
        k_dev = gemm_device(ln_out_dev, self.k_proj_w_dev[i])  # (seq, n_kv_head*d_k)
        v_dev = gemm_device(ln_out_dev, self.v_proj_w_dev[i])  # (seq, n_kv_head*d_k)
        
        # → GPU TRANSPOSE: (seq, n_heads*d_k) → (n_heads*seq, d_k)
        q_hm = to_head_major(q_dev, n_head, seq, d_k)
        k_hm = to_head_major(k_dev, n_kv_head, seq, d_k)
        v_hm = to_head_major(v_dev, n_kv_head, seq, d_k)
        
        # RoPE in-place (no change)
        apply_rope_device(q_hm, self.cos_dev, self.sin_dev, seq, prev_seq)
        apply_rope_device(k_hm, self.cos_dev, self.sin_dev, seq, prev_seq)
        
        if is_prefill:
            # → GPU CACHE WRITE: K, V into GPU DeviceTensor cache
            flat_to_cache(k_hm, cache["k"], n_kv_head, seq, d_k, max_seq, 0)
            flat_to_cache(v_hm, cache["v"], n_kv_head, seq, d_k, max_seq, 0)
            # → ATTENTION: K,V directly from head-major flat (matches attention input format)
            attn_dev = attention_gqa_device(q_hm, k_hm, v_hm, n_head, n_kv_head, causal=True)
        else:
            # → GPU CACHE WRITE: K, V into GPU DeviceTensor cache (at position prev_seq)
            flat_to_cache(k_hm, cache["k"], n_kv_head, seq, d_k, max_seq, prev_seq)
            flat_to_cache(v_hm, cache["v"], n_kv_head, seq, d_k, max_seq, prev_seq)
            # → GPU CACHE READ: K, V from GPU cache as head-major flat
            k_view = cache_to_flat(cache["k"], n_kv_head, total_after, d_k, max_seq)
            v_view = cache_to_flat(cache["v"], n_kv_head, total_after, d_k, max_seq)
            attn_dev = attention_gqa_device(q_hm, k_view, v_view, n_head, n_kv_head, causal=False)
        
        # → GPU TRANSPOSE BACK: (n_head*seq, d_k) → (seq, n_head*d_k)
        attn_sm = to_seq_major(attn_dev, n_head, seq, d_k)
        
        # Output projection + residual (no change)
        o_dev = gemm_device(attn_sm, self.o_proj_w_dev[i])
        h_dev = gpu.allocate((seq, n_embd), np.float32)
        add_device(o_dev, residual_dev, out_dev=h_dev)
        
        # MLP (no change — already GPU-resident, no reshape needed)
        # ...
    
    # Final norm + LM head (no change)
    # Single sync + to_host
    gpu.synchronize()
    return gpu.to_host(logits_dev).reshape(...)
```

The key change: **zero to_host/to_device calls in the layer loop.**

### 5. Expected Performance

- **TTFT**: 1330ms → ~150-250ms (eliminating ~240 DMA transfers in prefill path)
- **TPOT**: 73ms → ~15-30ms (eliminating decode DMA, but still 1 sync per forward call)
- **Throughput**: 10 tok/s → ~30-50 tok/s

### 6. Files to Modify

| File | Change |
|------|--------|
| `triton_llm/kernels/transpose_2d.py` | **NEW** — transpose + cache copy kernels |
| `triton_llm/kernels/__init__.py` | Export `transpose_2d` |
| `smollm2_triton/model.py` | Replace CPU to_host/to_device in layer loop with GPU transpose/cache ops |

### 7. Backward Compatibility

Phase 1 GPU functions unchanged. All existing CPU path tests unchanged. Phase 2 only modifies `_forward_cached_gpu()` and `_init_cache_gpu()`. `generate_gpu()` continues to work.

### 8. Correctness Verification

- All 110 tests still pass
- `test_gpu_resident_full_trip`: GPU vs CPU bit-exact (same as Phase 1)

### 9. Decode Path Optimization Details

For decode (seq=1), operations simplify:
- `gemm_device` of `(1, n_embd)` × `(n_embd, n_head*d_k)` → trivial
- `to_head_major` of `(1, n_head*d_k)` → `(n_head*1, d_k)` → just reshape, same memory layout! Can return a pointer view without copying.
  
  Actually for seq=1: `(1, n_head*d_k)` as head-major `(n_head, 1, d_k)` → `(n_head, d_k)` — the memory layout IS already contiguous per-head in the original. So `to_head_major` for seq=1 is a no-copy operation (just change the pointer stride interpretation). But since we need DeviceTensor, we'd still need a copy unless we create a "view" DeviceTensor.

  For simplicity in Phase 2, just do the copy (d_k=64, n_head=9, seq=1 → 576 floats = 2.3 KB, negligible).

- `cache_to_flat` and `flat_to_cache` for seq=1: same small copy cost.
