## Problem

Current architecture does ~15 CPU-to-GPU round-trips per layer per decode step (420+ per step for SmolLM2-135M 30 layers):

for each kernel:
  gpu.to_device(input)    -> CPU to GPU copy
  kernel_launch           -> ~10-50us Triton launch overhead
  gpu.synchronize()       -> stall, wait for GPU
  gpu.to_host(output)     -> GPU to CPU copy

Each round-trip adds ~30-70us overhead. For 420 calls x 100 decode tokens = ~1.2-3s just in overhead per generation.

## Root cause

Three independent but compounding issues:

### 1. Weights copied CPU to GPU on every call
Every gemm() call is (np.array, np.array) -> np.array. It to_device()s the weight matrix every time, even though weights never change between calls. They should be allocated on GPU once at init time and stay there.

### 2. Inter-kernel sync is unnecessary
Each kernel does synchronize() + to_host() + return. The calling code then immediately calls the next kernel with the result. The sync + copy-back + re-copy adds latency but zero value, the GPU could just pass device pointers directly.

### 3. KV cache lives on CPU
Cache is allocated as np.zeros() on CPU memory. Every decode step does:
  k_view = cache["k"][:, :total_after, :].reshape(-1, d_k)
which copies data to a new CPU buffer, then to_device() copies it again to GPU. The NOTE in model.py already marks this as a known issue.

## Proposed design

### Phase 1: GPU-resident weights with device-pointer passing
Add a GPUBuffer class that stores weights on GPU once at init time. Create device-pointer variants of kernel launchers (gemm, rms_norm, rope, attention_gqa) that skip to_device / to_host and return a device pointer instead.

### Phase 2: GPU KV cache
Allocate KV cache directly on GPU memory. Slice writes in decode use device-pointer arithmetic, no CPU copy.

### Phase 3: Fused decode loop
Single decode step runs entirely on GPU:
- embed token on GPU
- for each layer: all kernels pass device pointers
- only copy final logits back to CPU for sampling

### Expected impact

| Bottleneck          | Before       | After          |
|---------------------|-------------|----------------|
| Per-kernel CPU copy | 420x per step | 1x at end    |
| KV cache copy       | ~256KB/step  | 0             |
| Weight re-upload    | ~99% of calls | 0            |
| Est. TPOT (FP32)    | ~250ms       | ~5-15ms       |
| Est. TPOT (FP16)    |              | ~1-2ms        |

## Notes
- No new kernels needed, just device-pointer variants of existing ones
- FP16 is complementary (Phase 4), not required for the fix
