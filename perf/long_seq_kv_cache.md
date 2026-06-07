# Long-Sequence Performance: KV Cache vs Full Recompute

**System:** H100 80GB | CUDA 13.0 | Triton 3.4.0 | batch=1 | float32 | random weights

## Results

| case | prompt | gen | total_seq | full(s) | cache(s) | speedup | per_token full(ms) | per_token cache(ms) | quality |
|------|--------|-----|-----------|---------|----------|---------|-------------------|--------------------|---------|
| baseline | 8 | 10 | 18 | 1.83s | 1.76s | **1.04×** | 182.9 | 175.9 | ✅ |
| baseline | 8 | 50 | 58 | 9.42s | 8.83s | **1.07×** | 188.5 | 176.6 | ✅ |
| decode-heavy | 8 | 100 | 108 | 20.17s | 17.66s | **1.14×** | 201.7 | 176.6 | ✅ |
| decode-heavy | 8 | 200 | 208 | 45.87s | 35.39s | **1.30×** | 229.4 | 177.0 | ✅ |
| decode-heavy | 8 | 500 | 508 | 150.40s | 91.24s | **1.65×** | 300.8 | 182.5 | ✅ |
| prefill-heavy | 256 | 10 | 266 | 3.17s | 1.95s | **1.63×** | 317.3 | 194.7 | ✅ |
| prefill-heavy | 512 | 10 | 522 | 4.29s | 2.16s | **1.99×** | 429.3 | 215.7 | ✅ |
| balanced | 128 | 128 | 256 | 35.16s | 23.07s | **1.52×** | 274.6 | 180.2 | ✅ |
| balanced | 256 | 64 | 320 | 20.17s | 11.87s | **1.70×** | 315.2 | 185.5 | ✅ |
| near-limit | 510 | 10 | 520 | 4.31s | 2.13s | **2.02×** | 431.1 | 213.1 | ✅ |
| near-limit | 254 | 100 | 354 | 32.96s | 18.41s | **1.79×** | 329.6 | 184.1 | ✅ |

## Key Takeaways

### 1. Speedup scales with sequence length

As total sequence grows, KV cache advantage increases from **1.04× → 2.02×**:

| total_seq | speedup | regime |
|-----------|---------|--------|
| 18 | 1.04× | GEMM-dominated |
| 108 | 1.14× | |
| 256 | 1.52× | attention cost growing |
| 354 | 1.79× | |
| 508 | 1.65× | |
| 520 | **2.02×** | highest |

### 2. KV cache per-token latency is roughly constant (~175-180ms/token)

The KV cache path is **stable at ~175-180ms per token** regardless of gen length. This confirms the asymptotic O(T) behavior — each decode step does the same amount of work independent of total sequence length.

### 3. Full recompute per-token latency grows with sequence (O(T²))

Full recompute goes from **183ms/token** (gen=10) to **301ms/token** (gen=500) — a **1.65× increase** driven entirely by attention's quadratic cost. At gen=500 the forward pass processes a full 508-token sequence each step.

### 4. Prefill-heavy shows strong KV cache benefit (1.63-2.02×)

When prompt is long and gen is short, KV cache saves the expensive full prompt forward pass on every decode step — this is where the speedup is most dramatic.

### 5. Quality is perfect across all 11 configurations

All output-quality comparisons PASS — greedy decoding with KV cache produces identical results to full recompute.

## What This Means

The GPT-2 Small's KV cache proves the asymptotic O(T²) → O(T) improvement is real, but with a **cross-over point around total_seq=100** below which the benefit is <1.1×. For models with larger n_embd (Medium/Large/XL), the cross-over will shift right since GEMM is even more dominant.

**For a model where KV cache really shines**, we need either:
- Much longer sequences (4K-8K+) where attention dominates
- Cross-attention models (encoder-decoder) where the cache saves the entire encoder forward pass
- Multi-head attention with very high per-head dimension

All 11 quality checks PASS. See `scripts/benchmark_long_seq.py` for the benchmark harness.
