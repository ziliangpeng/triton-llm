# 🗺️ Triton-LLM — Complete Roadmap

**Repo:** [ziliangpeng/triton-llm](https://github.com/ziliangpeng/triton-llm)

**Goal:** Build production-quality LLM inference kernels in pure Python + Triton (zero PyTorch at runtime), supporting multiple model architectures.

---

## ✅ Phase 0 — Infrastructure (DONE)

| PR | What | Status |
|----|------|--------|
| #4 | GPU memory allocator (cudaMalloc + ctypes) | ✅ Merged |
| #7 | Cross-vendor GPU allocator (CUDA + HIP) | ✅ Merged |
| #27 | **Shared kernel library `triton_llm/kernels/`** — unified all 11 kernels from per-model dirs | ✅ Merged |

---

## ✅ Phase 1 — GPT-2 Stack (DONE — 19 PRs merged)

### M1 — Core Math Primitives
| Component | PR | Status |
|-----------|----|--------|
| `kernels/gemm.py` — tiled GEMM with k-loop | #8, #14 | ✅ |
| `kernels/add.py` — element-wise add | #11, #12 | ✅ |
| `kernels/gelu.py` — GELU activation | #11 | ✅ |
| `kernels/layernorm.py` — LayerNorm (2D/3D) | #9, #14 | ✅ |

### M2 — Attention Stack
| Component | PR | Status |
|-----------|----|--------|
| `kernels/softmax.py` — online softmax (NaN-safe) | #13 | ✅ |
| `kernels/attention.py` — fused Q×K→softmax→×V | #14 | ✅ |
| `kernels/embedding.py` — fused token + positional embedding | #15 | ✅ |

### M3 — Full Model & Generation
| Component | PR | Status |
|-----------|----|--------|
| `gpt2_triton/model.py` — GPT-2 wiring (12 layers) | #16 | ✅ |
| `gpt2_triton/config.py` — small/medium/large/xl | #16 | ✅ |
| `scripts/run_gpt2.py` — end-to-end CLI demo | #16 | ✅ |

### M4 — KV Cache
| Feature | PR | Status |
|---------|----|--------|
| Autoregressive KV-cached decode | #18 | ✅ |
| Pre-allocated KV cache (no `np.concatenate` in hot path) | #26 | ✅ |

### Benchmarks
| Benchmark | Status |
|-----------|--------|
| GPT-2 long-seq scalability (`perf/long_seq_kv_cache.md`) | ✅ Done |
| SmolLM2 KV cache comprehensive (15/17 cases, up to **4.85×** speedup) | ✅ Done |

---

## ✅ Phase 2 — SmolLM2 / Llama-family Stack (DONE)

### P1 — Llama Primitives
| Kernel | PR | Status |
|--------|----|--------|
| `kernels/rms_norm.py` — RMSNorm | #20 | ✅ |
| `kernels/rope.py` — Rotary Position Embedding | #21 | ✅ |
| `kernels/swiglu.py` — SwiGLU FFN | #22 | ✅ |
| `kernels/attention_gqa.py` — Grouped-Query Attention | #23 | ✅ |

### P2 — Model Integration & Verification
| Component | PR | Status |
|-----------|----|--------|
| `smollm2_triton/model.py` — full SmolLM2 wiring (30 layers, 3KV, GQA) | #24 | ✅ |
| `smollm2_triton/config.py` — SmolLM2 135M (supports 360M, 1.7B) | #24 | ✅ |
| Real-weight end-to-end correctness (cosine >0.999, top-5 100%) | #24 | ✅ |
| `scripts/run_smollm2.py` — CLI demo with HF model download | #24 | ✅ |
| Post-merge code quality review | #25 | ✅ |

### P3 — Multi-model Infrastructure
| Feature | Status |
|---------|--------|
| Shared `triton_llm/kernels/` — all 11 kernels in one place | #27 ✅ |
| Both models (GPT-2, SmolLM2) import from same kernel library | #27 ✅ |
| Cross-model inference tested on H100 (87/87 tests pass) | Verified ✅ |

---

## ⏳ Up Next — Recommended Order

### M2 — Batched GEMM (`kernels/gemm.py` → batch support)
**Why:** Current GEMM only supports M=1 batch. Adding batch dim enables simultaneous prompting, KV cache warm-up, and more realistic usage.
- Add batch dimension (B) to GEMM kernel
- Support batched `gemm(X @ W)` where X is `(B, M, K)`
- Update model callers for batch support
- Cost: 1-2 PRs, medium complexity

### M3 — Advanced Attention / KV Cache Optimization
**Options (pick one):**
- **Flash Attention in Triton** — fused online-attention with tiling, avoids materializing `(seq, seq)` attention matrix. Best for long-context. High complexity.
- **GQA KV slice optimization** — eliminate the CPU-side copy in pre-allocated KV cache by supporting custom head stride in the attention kernel. Medium complexity.
- **PageAttention-style paged KV cache** — for production serving. High complexity.

### M6 — Autotune & Architecture Specialization
**Why:** All kernels currently use hardcoded BLOCK_SIZE. `@triton.autotune` selects optimal configs per architecture.
- Add `@triton.autotune` to GEMM, Softmax, Attention kernels
- H100-specific vs AMD MI350X-specific config pools
- Benchmark-driven tuning
- Cost: medium

### P6 — Exploration Items (lower priority)
| Item | Why deferred |
|------|-------------|
| Triton autotune for batched GEMM | Depends on M2 first |
| Triton autotune for batched attention | Depends on M3 first |

---

## 🔮 Deferred (no concrete plan)

- FP16/mixed precision support
- Batch > 1 for inference (blocked on M2)
- Speculative decoding
- HTTP serving (`/v1/chat/completions`)
- Tensor parallelism
- More model species (Gemma, Qwen2.5, etc.)

---

## 📊 Current Stats

| Metric | Value |
|--------|-------|
| Kernels | 11 (all shared in `triton_llm/kernels/`) |
| Models | 2 (GPT-2, SmolLM2) |
| Test files | 16 (87 total tests on H100) |
| Merged PRs | 21 |
| Architectures | Conv1D (GPT-2), Llama GQA (SmolLM2) |
