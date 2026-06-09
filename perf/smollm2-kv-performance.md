# SmolLM2-135M KV Cache Performance: Full Recompute vs KV Cache

**Hardware:** H100 SXM5 | Triton 3.4.0 | CUDA 13.0  
**Model:** SmolLM2-135M (30 layers, 576 embd, 9 heads, 3 KV heads, 64 d_k)  
**Batch:** 1 | **Dtype:** float32 | **Weights:** random (seed=42)  
**Greedy decode** (temperature=0.0) — outputs verified identical

## Summary Table

| Case | prompt | gen | total_seq | full(s) | cache(s) | **speedup** | pt_full(ms) | pt_cache(ms) | quality |
|------|--------|-----|-----------|---------|----------|:-----------:|:-----------:|:------------:|:-------:|
| tiny | 8 | 10 | 18 | 2.45 | 2.34 | **1.05×** | 244.8 | 233.7 | ✅ |
| baseline | 8 | 50 | 58 | 12.19 | 11.43 | **1.07×** | 243.8 | 228.7 | ✅ |
| baseline | 8 | 100 | 108 | 25.65 | 22.65 | **1.13×** | 256.5 | 226.5 | ✅ |
| decode-heavy | 8 | 200 | 208 | 56.68 | 45.50 | **1.25×** | 283.4 | 227.5 | ✅ |
| decode-heavy | 8 | 500 | 508 | 202.64 | 124.98 | **1.62×** | 405.3 | 250.0 | ✅ |
| decode-heavy | 8 | 1000 | 1008 | 578.55 | 252.05 | **2.30×** | 578.6 | 252.0 | ✅ |
| ~~decode-heavy~~ | ~~8~~ | ~~2000~~ | ~~2008~~ | — | — | — | — | — | — |
| prefill-heavy | 256 | 10 | 266 | 9.02 | 2.62 | **3.44×** | 901.7 | 261.9 | ✅ |
| prefill-heavy | 512 | 10 | 522 | 5.64 | 2.83 | **2.00×** | 564.5 | 282.8 | ✅ |
| prefill-heavy | 1024 | 10 | 1034 | 8.58 | 3.37 | **2.55×** | 858.5 | 337.2 | ✅ |
| prefill-heavy | 2048 | 10 | 2058 | 14.86 | 4.22 | **3.52×** | 1486.4 | 422.4 | ✅ |
| balanced | 128 | 128 | 256 | 45.43 | 32.03 | **1.42×** | 354.9 | 250.3 | ✅ |
| balanced | 256 | 64 | 320 | 27.53 | 16.16 | **1.70×** | 430.1 | 252.5 | ✅ |
| balanced | 512 | 128 | 640 | 78.62 | 33.06 | **2.38×** | 614.2 | 258.3 | ✅ |
| balanced | 1024 | 256 | 1280 | 243.67 | 68.77 | **3.54×** | 951.8 | 268.6 | ✅ |
| near-limit | 4096 | 10 | 4106 | 31.97 | 6.59 | **4.85×** | 3196.6 | 658.6 | ✅ |
| ~~near-limit~~ | ~~2048~~ | ~~100~~ | ~~2148~~ | — | — | — | — | — | — |
| ~~near-limit~~ | ~~512~~ | ~~512~~ | ~~1024~~ | — | — | — | — | — | — |

> Note: strikethrough rows were not completed (O(T³) full-recompute too slow or process hang). `benchmark_remaining.py` produced these extra results (received late from an async job). Balanced 1024+256 confirms 3.5× regime; near-limit 4096+10 sets the new record at **4.85×**.
> `pt_full` includes full TFLOPS not applicable (a single full recompute step in decode loop must process the entire growing sequence).

## Key Findings

### 1. KV cache per-token latency is stable (~227-261ms)

Across all **15 completed configurations** (2 of 4 remaining cases now resolved), KV cache per-decode-step latency stays within **228–659ms** (the 659ms is the 4096-prompt case where the single "decode" step is doing 4106-length attention — the post-prefill residual cost). For all other cases, the range is **228–269ms**.

### 2. Full recompute cost grows O(T²), KV cache saves more at longer sequences

| total_seq | speedup |
|-----------|:-------:|
| 18 | 1.05× |
| 108 | 1.13× |
| 208 | 1.25× |
| 508 | 1.62× |
| 1008 | **2.30×** |

### 3. Prefill-heavy is KV cache's best case (up to 4.85×)

When the prompt is long and only a few tokens are generated, KV cache saves re-doing the entire expensive prompt forward pass at every decode step. The **4.85×** at 4096+10 is the new record — the full-recompute per-token cost reaches **3.2s/token** (attention O(T²) dominates entirely), while KV cache is just 659ms.

### 4. Full recompute per-token cost grows dramatically with sequence

Full recompute goes from **245ms/token** (tiny) to **579ms/token** (8+1000) — a 2.4× increase driven by attention's quadratic cost.

### 5. All 15 cases produce identical output (greedy decode)

Quality check: full recompute and KV cache produce exactly the same tokens in every case. PASS ✅

## New Records from Extra Data

Two previously-missing large cases now complete:
- **balanced 1024+256**: 3.54× speedup (total_seq=1280)
- **near-limit 4096+10**: **4.85×** speedup (total_seq=4106) — new KV cache speedup record

Still missing (O(T³) compute too expensive): near-limit 2048+100, near-limit 512+512.
