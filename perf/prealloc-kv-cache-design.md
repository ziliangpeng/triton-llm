# Pre-allocated KV Cache — Design

## Motivation

Current KV cache allocates `(n_kv_head, 0, d_k)` on init, then does
`np.concatenate` on every decode step. This causes O(T²) memory
allocation overhead as seq grows.

For SmolLM2-135M with max_seq=8192:
- n_kv_head=3, d_k=64 → per-layer cache = 3 × 8192 × 64 × 4 bytes = **6 MB/layer**
- 30 layers → **180 MB total** → 0.22% of H100 80 GB
- Pre-allocating the full max_seq is essentially free.

## Design

### `_init_cache(max_seq=None)`

```
if max_seq is None: max_seq = config.max_position_embeddings

self.kv_cache = [{
    "k": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
    "v": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
} for _ in range(n_layer)]
self._cache_len = 0   # number of populated positions (always <= max_seq)
```

### Prefill (prev_seq=0, seq=N)

```
cache["k"][:, 0:seq, :] = k_rope.reshape(n_kv_head, seq, d_k)
cache["v"][:, 0:seq, :] = v_flat.reshape(n_kv_head, seq, d_k)
self._cache_len = seq
```

### Decode (prev_seq>0, seq=1)

```
cache["k"][:, prev_seq, :] = k_rope.reshape(n_kv_head, d_k)   # single position
cache["v"][:, prev_seq, :] = v_flat.reshape(n_kv_head, d_k)
self._cache_len = prev_seq + 1
```

### Attention input

```
cache["k"][:, :self._cache_len, :].reshape(-1, d_k)  # (n_kv_head * _cache_len, d_k)
cache["v"][:, :self._cache_len, :].reshape(-1, d_k)
```

### Effects

- No `np.concatenate` anywhere in the hot path
- Zero allocation after `_init_cache`
- Cache length is tracked by `_cache_len`, not `cache["k"].shape[1]`
- **`prev_seq`** is still derived from `_cache_len` before the current step
- `generate()` path unchanged — same API

### Changes needed

1. `_init_cache()` — allocate full arrays, add `_cache_len`
2. `_forward_cached()` — replace concat with slice writes, use `_cache_len` for attention
3. `generate()` — no change needed (already calls `_init_cache` then `_forward_cached`)
4. Tests — `test_model_kv_cache_equivalence`, `test_model_kv_cache_decode` should pass unchanged
5. Add `test_model_prealloc_cache_full_trip` — prefill + multi-decode, compare to full forward

### Verification

- KV cache equivalence: `forward(full_seq, use_cache=False)` vs `init_cache → forward(prompt, use_cache=True) → forward(next, use_cache=True) → ...`
- Token equivalence: greedy generate with pre-alloc cache matches full-recompute
- Benchmarks: measure per-step timing vs current concat-based cache
