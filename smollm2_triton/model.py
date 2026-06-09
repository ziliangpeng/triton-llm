"""SmolLM2 (Llama-architecture) inference model with KV cache.

Wires RMSNorm, RoPE, SwiGLU, and GQA attention Triton kernels into
a complete autoregressive language model.  Batch=1 only.
"""

import time

import numpy as np

from triton_llm import gpu
from triton_llm.kernels.gemm import gemm, gemm_device
from triton_llm.kernels.add import add, add_device
from triton_llm.kernels.rms_norm import rms_norm, rms_norm_device
from triton_llm.kernels.rope import precompute_cos_sin, precompute_cos_sin_device, apply_rope, apply_rope_device
from triton_llm.kernels.swiglu import swiglu, swiglu_device
from triton_llm.kernels.attention_gqa import attention_gqa, attention_gqa_device
from triton_llm.kernels.transpose_2d import (
    to_head_major_dev,
    to_seq_major_dev,
    flat_to_cache_dev,
    cache_to_flat_dev,
)

from smollm2_triton.config import SmolLM2Config


class SmolLM2ForCausalLM:
    """SmolLM2 (Llama-architecture) inference model backed by Triton kernels.

    Parameters
    ----------
    config : SmolLM2Config
        Model configuration.
    weights : dict of str -> np.ndarray
        Dictionary mapping HuggingFace weight names (e.g.
        ``model.layers.0.self_attn.q_proj.weight``) to NumPy arrays.
        Linear weights are expected in HF format (out_features, in_features)
        and are transposed to (in_features, out_features) at load time.
    """

    def __init__(self, config: SmolLM2Config, weights: dict):
        self.config = config
        n_layer = config.n_layer
        n_embd = config.n_embd
        n_head = config.n_head
        n_kv_head = config.n_kv_head
        d_k = n_embd // n_head

        if n_head % n_kv_head != 0:
            raise ValueError(
                f"n_head ({n_head}) must be divisible by n_kv_head ({n_kv_head})"
            )

        # --- Token embedding (no positional embedding for Llama) ---
        self.wte = np.require(
            weights["model.embed_tokens.weight"],
            dtype=np.float32,
            requirements=["C_CONTIGUOUS"],
        )

        # --- Final RMSNorm ---
        self.ln_f_g = np.require(
            weights["model.norm.weight"],
            dtype=np.float32,
            requirements=["C_CONTIGUOUS"],
        )

        # --- RoPE precomputation ---
        cos, sin = precompute_cos_sin(
            config.max_position_embeddings, d_k, theta=config.rope_theta
        )
        # Keep on CPU for slicing; Triton kernels transfer to GPU internally.
        self.cos = cos  # (max_seq, d_k // 2)
        self.sin = sin

        # --- LM head ---
        # Llama ties lm_head with embed_tokens when tie_word_embeddings=True
        if "lm_head.weight" in weights:
            lm_w = weights["lm_head.weight"]
        elif config.tie_word_embeddings:
            lm_w = weights["model.embed_tokens.weight"]
        else:
            raise KeyError("lm_head.weight is missing from weights and tie_word_embeddings is False")
        # Transpose: (vocab, hidden) -> (hidden, vocab) for gemm(hidden, w)
        self.lm_head_w = np.require(
            lm_w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )

        # --- Per-layer weight lists ---
        self.ln_1_g: list[np.ndarray] = []
        self.ln_2_g: list[np.ndarray] = []
        self.q_proj_w: list[np.ndarray] = []
        self.k_proj_w: list[np.ndarray] = []
        self.v_proj_w: list[np.ndarray] = []
        self.o_proj_w: list[np.ndarray] = []
        self.gate_proj_w: list[np.ndarray] = []
        self.up_proj_w: list[np.ndarray] = []
        self.down_proj_w: list[np.ndarray] = []

        for i in range(n_layer):
            # --- Attention QKV projections ---
            # HF stores Linear weights as (out_features, in_features).
            # gemm(hidden, w) expects (in, out), so transpose.
            w = weights[f"model.layers.{i}.self_attn.q_proj.weight"]
            self.q_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            w = weights[f"model.layers.{i}.self_attn.k_proj.weight"]
            self.k_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            w = weights[f"model.layers.{i}.self_attn.v_proj.weight"]
            self.v_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )

            # --- Attention output projection ---
            w = weights[f"model.layers.{i}.self_attn.o_proj.weight"]
            self.o_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )

            # --- MLP projections (SwiGLU needs gate + up) ---
            w = weights[f"model.layers.{i}.mlp.gate_proj.weight"]
            self.gate_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            w = weights[f"model.layers.{i}.mlp.up_proj.weight"]
            self.up_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            w = weights[f"model.layers.{i}.mlp.down_proj.weight"]
            self.down_proj_w.append(
                np.require(w.T.copy(), dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )

            # --- Layer norms (RMSNorm — no bias) ---
            self.ln_1_g.append(
                np.require(
                    weights[f"model.layers.{i}.input_layernorm.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_2_g.append(
                np.require(
                    weights[f"model.layers.{i}.post_attention_layernorm.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

        # --- GPU-resident weight storage (Phase 1: EC32) ---
        # Lazily initialized by _init_gpu_weights() — kept as empty lists so
        # CPU-only workloads (CPU-based tests) don't need a GPU runtime.
        self._gpu_initialized = False
        self.q_proj_w_dev: list[gpu.DeviceTensor] = []
        self.k_proj_w_dev: list[gpu.DeviceTensor] = []
        self.v_proj_w_dev: list[gpu.DeviceTensor] = []
        self.o_proj_w_dev: list[gpu.DeviceTensor] = []
        self.gate_proj_w_dev: list[gpu.DeviceTensor] = []
        self.up_proj_w_dev: list[gpu.DeviceTensor] = []
        self.down_proj_w_dev: list[gpu.DeviceTensor] = []
        self.ln_1_w_dev: list[gpu.DeviceTensor] = []
        self.ln_2_w_dev: list[gpu.DeviceTensor] = []
        self.ln_f_w_dev: gpu.DeviceTensor | None = None
        self.lm_head_w_dev: gpu.DeviceTensor | None = None
        self.cos_dev: gpu.DeviceTensor | None = None
        self.sin_dev: gpu.DeviceTensor | None = None

    # ------------------------------------------------------------------
    # GPU weights (lazy init)
    # ------------------------------------------------------------------

    def _init_gpu_weights(self):
        """Lazily copy all weights to GPU (called on first GPU forward pass)."""
        if self._gpu_initialized:
            return
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.q_proj_w_dev.append(gpu.to_device(self.q_proj_w[i]))
            self.k_proj_w_dev.append(gpu.to_device(self.k_proj_w[i]))
            self.v_proj_w_dev.append(gpu.to_device(self.v_proj_w[i]))
            self.o_proj_w_dev.append(gpu.to_device(self.o_proj_w[i]))
            self.gate_proj_w_dev.append(gpu.to_device(self.gate_proj_w[i]))
            self.up_proj_w_dev.append(gpu.to_device(self.up_proj_w[i]))
            self.down_proj_w_dev.append(gpu.to_device(self.down_proj_w[i]))
            self.ln_1_w_dev.append(gpu.to_device(self.ln_1_g[i]))
            self.ln_2_w_dev.append(gpu.to_device(self.ln_2_g[i]))
        self.ln_f_w_dev = gpu.to_device(self.ln_f_g)
        self.lm_head_w_dev = gpu.to_device(self.lm_head_w)
        d_k = self.config.n_embd // self.config.n_head
        self.cos_dev, self.sin_dev = precompute_cos_sin_device(
            self.config.max_position_embeddings, d_k, theta=self.config.rope_theta
        )
        self._gpu_initialized = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """Run a forward pass over input tokens.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Input token IDs.
        use_cache : bool
            If True, use KV cache for incremental attention.

        Returns
        -------
        logits : np.ndarray, shape ``(1, seq, vocab_size)``, float32
            Unnormalised logits for each position.
        """
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
            raise ValueError(
                f"token_ids must be a non-empty 2D array with shape (1, seq), "
                f"got shape {token_ids.shape}"
            )
        if use_cache:
            return self._forward_cached(token_ids)
        return self._forward_full(token_ids)

    def _forward_full(self, token_ids: np.ndarray) -> np.ndarray:
        """Full forward pass over *all* input tokens (no KV cache)."""
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd
        seq = token_ids.shape[1]
        if seq > config.max_position_embeddings:
            raise ValueError(
                f"Sequence length {seq} exceeds max_position_embeddings ({config.max_position_embeddings})"
            )

        # --- Token embedding (no positional embedding for Llama) ---
        hidden = self._embed(token_ids)  # (1, seq, n_embd)

        # --- Transformer layers ---
        for i in range(n_layer):
            # --- Attention sub-block ---
            residual = hidden  # (1, seq, n_embd)
            h = hidden.reshape(-1, n_embd)  # (seq, n_embd)
            h = rms_norm(h, self.ln_1_g[i], config.rms_norm_eps)
            h = self._apply_attention(h, i, prev_seq=0)  # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)  # (1, seq, n_embd)

            # --- MLP sub-block ---
            residual = hidden
            h = hidden.reshape(-1, n_embd)
            h = rms_norm(h, self.ln_2_g[i], config.rms_norm_eps)
            h = self._apply_mlp(h, i)  # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)

        # --- Final RMSNorm ---
        h = hidden.reshape(-1, n_embd)
        h = rms_norm(h, self.ln_f_g, config.rms_norm_eps)

        # --- LM head ---
        logits = gemm(h, self.lm_head_w)  # (seq, n_embd) @ (n_embd, vocab) -> (seq, vocab)
        return logits.reshape(1, seq, config.vocab_size)

    # ------------------------------------------------------------------
    # KV-cached forward
    # ------------------------------------------------------------------

    def _init_cache(self, max_seq: int | None = None):
        """Initialise pre-allocated KV cache for generation.

        Allocates the full ``(n_kv_head, max_seq, d_k)`` arrays to
        eliminate ``np.concatenate`` at every decode step.

        Parameters
        ----------
        max_seq : int or None
            Maximum sequence length to pre-allocate. Defaults to
            ``config.max_position_embeddings``.
        """
        if max_seq is None:
            max_seq = self.config.max_position_embeddings
        elif max_seq <= 0:
            raise ValueError(
                f"max_seq must be a positive integer, got {max_seq}"
            )
        elif max_seq > self.config.max_position_embeddings:
            raise ValueError(
                f"max_seq ({max_seq}) cannot exceed model's "
                f"max_position_embeddings ({self.config.max_position_embeddings})"
            )
        n_layer = self.config.n_layer
        n_kv_head = self.config.n_kv_head
        d_k = self.config.n_embd // self.config.n_head
        self._cache_len = 0
        self.kv_cache = [
            {
                "k": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
                "v": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
            }
            for _ in range(n_layer)
        ]

    def _forward_cached(self, token_ids: np.ndarray) -> np.ndarray:
        """Incremental forward pass using KV cache.

        Supports two modes:
        - Prefill (cache is empty): process all tokens, store K/V.
        - Decode (cache has data): process a single token using
          cached K/V from previous steps.

        Must call ``_init_cache()`` before the first call.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd
        n_head = config.n_head
        n_kv_head = config.n_kv_head
        d_k = n_embd // n_head

        if not hasattr(self, "kv_cache") or self.kv_cache is None:
            raise RuntimeError("_init_cache() must be called before _forward_cached()")

        # Position offset = total tokens cached so far (_cache_len tracks seq dim)
        prev_seq = self._cache_len
        seq = token_ids.shape[1]

        is_prefill = (prev_seq == 0)
        if not is_prefill and seq > 1:
            raise ValueError(
                f"Decode mode requires seq=1, got seq={seq}. "
                "Use _forward_full() for multi-token forward passes "
                "when the cache is non-empty."
            )
        total_after = prev_seq + seq
        max_seq = self.kv_cache[0]["k"].shape[1]
        if total_after > max_seq:
            raise ValueError(
                f"Total sequence length {total_after} exceeds "
                f"pre-allocated cache size ({max_seq})"
            )

        # --- Token embedding ---
        hidden = self._embed(token_ids)  # (1, seq, n_embd)
        hidden = hidden.reshape(-1, n_embd)  # (seq, n_embd)

        is_prefill = (prev_seq == 0)

        for i in range(n_layer):
            cache = self.kv_cache[i]

            # --- Attention sub-block ---
            residual = hidden
            h = rms_norm(hidden, self.ln_1_g[i], config.rms_norm_eps)

            # QKV projections
            q = gemm(h, self.q_proj_w[i])  # (seq, n_head * d_k)
            k = gemm(h, self.k_proj_w[i])  # (seq, n_kv_head * d_k)
            v = gemm(h, self.v_proj_w[i])  # (seq, n_kv_head * d_k)

            # Reshape and transpose to head-major flat: (seq, n_head, d_k) -> (n_head, seq, d_k) -> (n_head * seq, d_k)
            q_flat = q.reshape(seq, n_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)
            k_flat = k.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)
            v_flat = v.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)

            # Apply RoPE
            cos_slice = self.cos[prev_seq:prev_seq + seq, :]
            sin_slice = self.sin[prev_seq:prev_seq + seq, :]
            q_rope = apply_rope(q_flat, cos_slice, sin_slice, seq_len=seq)
            k_rope = apply_rope(k_flat, cos_slice, sin_slice, seq_len=seq)

            if is_prefill:
                # Prefill: store K, V as slice into pre-allocated cache
                cache["k"][:, :seq, :] = k_rope.reshape(n_kv_head, seq, d_k)
                cache["v"][:, :seq, :] = v_flat.reshape(n_kv_head, seq, d_k)
                attn_out = attention_gqa(
                    q_rope, k_rope, v_flat, n_head, n_kv_head, causal=True
                )
            else:
                # Decode: write new K, V into the next position(s)
                k_3d = k_rope.reshape(n_kv_head, seq, d_k)
                v_3d = v_flat.reshape(n_kv_head, seq, d_k)
                cache["k"][:, prev_seq:prev_seq + seq, :] = k_3d
                cache["v"][:, prev_seq:prev_seq + seq, :] = v_3d
                # View of populated cache entries for attention
                # NOTE: Slicing + reshape forces a CPU-side copy (non-contiguous after
                # sequence-dim slice). Fixing this requires Triton GQA kernel to accept
                # custom head strides (kv_head_stride = max_seq * d_k).
                k_view = cache["k"][:, :total_after, :].reshape(-1, d_k)
                v_view = cache["v"][:, :total_after, :].reshape(-1, d_k)
                attn_out = attention_gqa(
                    q_rope, k_view, v_view,
                    n_head, n_kv_head,
                    causal=False,
                )

            # Output projection — attn_out from GQA is head-major flat
            # (n_head * seq, d_k), transpose back to (seq, n_head * d_k)
            attn_out_head_major = attn_out  # (n_head * seq, d_k)
            attn_out_seq_major = attn_out_head_major.reshape(n_head, seq, d_k).transpose(1, 0, 2).reshape(seq, n_head * d_k)
            out = gemm(attn_out_seq_major, self.o_proj_w[i])  # (seq, n_embd)
            hidden = add(out, residual)  # (seq, n_embd)

            # --- MLP sub-block ---
            residual = hidden
            h = rms_norm(hidden, self.ln_2_g[i], config.rms_norm_eps)
            h = self._apply_mlp(h, i)
            hidden = add(h, residual)

        # Update cache length after processing this step
        self._cache_len = total_after

        # --- Final RMSNorm ---
        h = rms_norm(hidden, self.ln_f_g, config.rms_norm_eps)

        # --- LM head ---
        logits = gemm(h, self.lm_head_w)  # (seq, vocab_size)
        return logits.reshape(1, seq, config.vocab_size)

    # ------------------------------------------------------------------
    # GPU-resident forward pass (Phase 1 + Phase 2)
    # ------------------------------------------------------------------

    def _init_cache_gpu(self, max_seq: int | None = None):
        """Initialise pre-allocated KV cache for GPU-resident path (Phase 2).

        Allocates DeviceTensors for KV cache storage on GPU, removing all
        CPU round-trips for cache read/write.

        Parameters
        ----------
        max_seq : int or None
            Maximum sequence length to pre-allocate. Defaults to
            ``config.max_position_embeddings``.
        """
        if max_seq is None:
            max_seq = self.config.max_position_embeddings
        elif max_seq <= 0:
            raise ValueError(
                f"max_seq must be a positive integer, got {max_seq}"
            )
        elif max_seq > self.config.max_position_embeddings:
            raise ValueError(
                f"max_seq ({max_seq}) cannot exceed model's "
                f"max_position_embeddings ({self.config.max_position_embeddings})"
            )
        n_layer = self.config.n_layer
        n_kv_head = self.config.n_kv_head
        d_k = self.config.n_embd // self.config.n_head
        self._cache_len = 0
        self._cache_max_seq = max_seq
        # Allocate zero-initialized GPU DeviceTensors (one-time copy at init)
        cache_np = np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32)
        self.kv_cache_dev = [
            {
                "k": gpu.to_device(cache_np.copy()),
                "v": gpu.to_device(cache_np.copy()),
            }
            for _ in range(n_layer)
        ]

    def _forward_cached_gpu(self, token_ids: np.ndarray) -> np.ndarray:
        """GPU-resident incremental forward pass using GPU KV cache (Phase 2).

        Fully GPU-resident: no to_host/to_device calls in the layer loop.
        Transpose and KV cache operations are done entirely on GPU via
        dedicated Triton kernels.

        Only one ``synchronize()`` and one ``to_host()`` at the end.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd
        n_head = config.n_head
        n_kv_head = config.n_kv_head
        d_k = n_embd // n_head

        self._init_gpu_weights()

        if not hasattr(self, "kv_cache_dev") or self.kv_cache_dev is None:
            raise RuntimeError("_init_cache_gpu() must be called before _forward_cached_gpu()")

        prev_seq = self._cache_len
        seq = token_ids.shape[1]

        is_prefill = (prev_seq == 0)
        if not is_prefill and seq > 1:
            raise ValueError(
                f"Decode mode requires seq=1, got seq={seq}. "
            )
        total_after = prev_seq + seq
        max_seq = self._cache_max_seq if hasattr(self, '_cache_max_seq') else self.kv_cache_dev[0]["k"].shape[1]
        if total_after > max_seq:
            raise ValueError(
                f"Total sequence length {total_after} exceeds "
                f"pre-allocated cache size ({max_seq})"
            )

        # --- Token embedding on CPU, then copy to GPU once ---
        hidden = self._embed(token_ids)  # (1, seq, n_embd) on CPU
        h_dev = gpu.to_device(hidden.reshape(-1, n_embd).copy())  # (seq, n_embd) on GPU

        for i in range(n_layer):
            cache = self.kv_cache_dev[i]

            # --- Attention sub-block ---
            residual_dev = h_dev

            # RMSNorm
            ln_out_dev = gpu.allocate((seq, n_embd), np.float32)
            rms_norm_device(h_dev, self.ln_1_w_dev[i], ln_out_dev, config.rms_norm_eps)

            # QKV projections on GPU
            q_dev = gemm_device(ln_out_dev, self.q_proj_w_dev[i])  # (seq, n_head * d_k)
            k_dev = gemm_device(ln_out_dev, self.k_proj_w_dev[i])  # (seq, n_kv_head * d_k)
            v_dev = gemm_device(ln_out_dev, self.v_proj_w_dev[i])  # (seq, n_kv_head * d_k)

            # GPU transpose: (seq, n_heads*d_k) -> (n_heads*seq, d_k)
            q_hm = to_head_major_dev(q_dev, n_head, seq, d_k)
            k_hm = to_head_major_dev(k_dev, n_kv_head, seq, d_k)
            v_hm = to_head_major_dev(v_dev, n_kv_head, seq, d_k)

            # Apply RoPE on GPU (in-place on q_hm and k_hm)
            apply_rope_device(q_hm, self.cos_dev, self.sin_dev, seq_len=seq, position_offset=prev_seq)
            apply_rope_device(k_hm, self.cos_dev, self.sin_dev, seq_len=seq, position_offset=prev_seq)

            if is_prefill:
                # GPU cache write: directly into DeviceTensor cache
                flat_to_cache_dev(k_hm, cache["k"], n_kv_head, seq, d_k, max_seq, 0)
                flat_to_cache_dev(v_hm, cache["v"], n_kv_head, seq, d_k, max_seq, 0)

                attn_dev = attention_gqa_device(
                    q_hm, k_hm, v_hm, n_head, n_kv_head, causal=True
                )
            else:
                # GPU cache write at position prev_seq
                flat_to_cache_dev(k_hm, cache["k"], n_kv_head, seq, d_k, max_seq, prev_seq)
                flat_to_cache_dev(v_hm, cache["v"], n_kv_head, seq, d_k, max_seq, prev_seq)

                # GPU cache read: populate cache slice as head-major flat
                k_view = cache_to_flat_dev(cache["k"], n_kv_head, total_after, d_k, max_seq)
                v_view = cache_to_flat_dev(cache["v"], n_kv_head, total_after, d_k, max_seq)

                attn_dev = attention_gqa_device(
                    q_hm, k_view, v_view,
                    n_head, n_kv_head,
                    causal=False,
                )

            # GPU transpose back: (n_head*seq, d_k) -> (seq, n_head*d_k)
            attn_sm = to_seq_major_dev(attn_dev, n_head, seq, d_k)

            # Output projection + residual on GPU
            o_dev = gemm_device(attn_sm, self.o_proj_w_dev[i])  # (seq, n_embd)
            h_dev = gpu.allocate((seq, n_embd), np.float32)
            add_device(o_dev, residual_dev, out_dev=h_dev)

            # --- MLP sub-block ---
            residual_dev = h_dev
            ln_out_dev = gpu.allocate((seq, n_embd), np.float32)
            rms_norm_device(h_dev, self.ln_2_w_dev[i], ln_out_dev, config.rms_norm_eps)

            gate_dev = gemm_device(ln_out_dev, self.gate_proj_w_dev[i])  # (seq, intermediate_size)
            up_dev = gemm_device(ln_out_dev, self.up_proj_w_dev[i])      # (seq, intermediate_size)
            act_dev = swiglu_device(gate_dev, up_dev)                     # (seq, intermediate_size)
            down_dev = gemm_device(act_dev, self.down_proj_w_dev[i])      # (seq, n_embd)
            h_dev = gpu.allocate((seq, n_embd), np.float32)
            add_device(down_dev, residual_dev, out_dev=h_dev)

        # Update cache length after processing this step
        self._cache_len = total_after

        # --- Final RMSNorm on GPU ---
        ln_out_dev = gpu.allocate((seq, n_embd), np.float32)
        rms_norm_device(h_dev, self.ln_f_w_dev, ln_out_dev, config.rms_norm_eps)

        # --- LM head on GPU ---
        logits_dev = gemm_device(ln_out_dev, self.lm_head_w_dev)  # (seq, vocab_size)

        # Single sync + to_host at the very end
        gpu.synchronize()
        return gpu.to_host(logits_dev).reshape(1, seq, config.vocab_size)

    def forward_gpu(self, token_ids: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """GPU-resident forward pass (dispatches to GPU cached path if cache is initialized).

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
        use_cache : bool
            If True, use GPU KV cache. ``_init_cache_gpu()`` must have been called.

        Returns
        -------
        logits : np.ndarray, shape ``(1, seq, vocab_size)``, float32
        """
        if token_ids.ndim != 2 or token_ids.shape[0] != 1 or token_ids.shape[1] == 0:
            raise ValueError(
                f"token_ids must be a non-empty 2D array with shape (1, seq), "
                f"got shape {token_ids.shape}"
            )
        if use_cache:
            return self._forward_cached_gpu(token_ids)
        return self._forward_full(token_ids)  # Falls back to CPU path for non-cached

    def generate_gpu(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> np.ndarray:
        """GPU-resident autoregressive generation.

        Uses GPU KV cache and GPU-resident forward pass for the entire
        prefill + decode loop. Only syncs once per forward call.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Prompt token IDs.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature. 0 = greedy.
        top_k : int
            If > 0, restrict sampling to the top-k most probable tokens.

        Returns
        -------
        tokens : np.ndarray, shape ``(1, seq + max_new_tokens)``, int32
        """
        if token_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {token_ids.shape[0]}")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must have at least 1 token")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        total_tokens = token_ids.shape[1] + max_new_tokens
        if total_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"total tokens ({total_tokens} = {token_ids.shape[1]} prompt "
                f"+ {max_new_tokens} new) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )

        self._init_cache_gpu()

        # Prefill
        logits = self._forward_cached_gpu(token_ids)

        tokens = token_ids.copy()
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :]
            next_token = self._sample(next_logits, temperature, top_k)
            tokens = np.concatenate(
                [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
            )
            if step < max_new_tokens - 1:
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                logits = self._forward_cached_gpu(new_token_arr)
        return tokens

    def _apply_attention(
        self, h_2d: np.ndarray, layer_idx: int, prev_seq: int = 0
    ) -> np.ndarray:
        """QKV projection -> reshape -> RoPE -> GQA -> output projection.

        Parameters
        ----------
        h_2d : np.ndarray, shape ``(seq, n_embd)``, float32
            Pre-normalised hidden states.
        layer_idx : int
            Transformer layer index.
        prev_seq : int
            Number of tokens cached before this sequence (for RoPE offset).

        Returns
        -------
        out : np.ndarray, shape ``(seq, n_embd)``, float32
            Attention output (before residual addition).
        """
        config = self.config
        n_embd = config.n_embd
        n_head = config.n_head
        n_kv_head = config.n_kv_head
        d_k = n_embd // n_head
        seq = h_2d.shape[0]

        # QKV projections
        q = gemm(h_2d, self.q_proj_w[layer_idx])  # (seq, n_head * d_k)
        k = gemm(h_2d, self.k_proj_w[layer_idx])  # (seq, n_kv_head * d_k)
        v = gemm(h_2d, self.v_proj_w[layer_idx])  # (seq, n_kv_head * d_k)

        # Reshape and transpose to head-major flat: (seq, n_head, d_k) -> (n_head, seq, d_k) -> (n_head * seq, d_k)
        q_flat = q.reshape(seq, n_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)
        k_flat = k.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)
        v_flat = v.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k)

        # Apply RoPE
        cos_slice = self.cos[prev_seq:prev_seq + seq, :]
        sin_slice = self.sin[prev_seq:prev_seq + seq, :]
        q_rope = apply_rope(q_flat, cos_slice, sin_slice, seq_len=seq)
        k_rope = apply_rope(k_flat, cos_slice, sin_slice, seq_len=seq)

        # GQA attention
        attn_out = attention_gqa(q_rope, k_rope, v_flat, n_head, n_kv_head, causal=True)

        # Output projection — attn_out from GQA is head-major flat (n_head * seq, d_k),
        # need to transpose back to (seq, n_head * d_k) for output projection
        attn_out_head_major = attn_out  # (n_head * seq, d_k)
        attn_out_seq_major = attn_out_head_major.reshape(n_head, seq, d_k).transpose(1, 0, 2).reshape(seq, n_head * d_k)
        out = gemm(attn_out_seq_major, self.o_proj_w[layer_idx])  # (seq, n_embd)
        return out

    # ------------------------------------------------------------------
    # MLP sub-block
    # ------------------------------------------------------------------

    def _apply_mlp(self, h_2d: np.ndarray, layer_idx: int) -> np.ndarray:
        """gate_proj -> up_proj -> SwiGLU -> down_proj.

        Parameters
        ----------
        h_2d : np.ndarray, shape ``(seq, n_embd)``, float32
        layer_idx : int
            Transformer layer index.

        Returns
        -------
        out : np.ndarray, shape ``(seq, n_embd)``, float32
        """
        # gate_proj: (seq, n_embd) -> (seq, intermediate_size)
        gate = gemm(h_2d, self.gate_proj_w[layer_idx])
        # up_proj: (seq, n_embd) -> (seq, intermediate_size)
        up = gemm(h_2d, self.up_proj_w[layer_idx])

        # SwiGLU activation
        h = swiglu(gate, up)  # (seq, intermediate_size)

        # down_proj: (seq, intermediate_size) -> (seq, n_embd)
        out = gemm(h, self.down_proj_w[layer_idx])
        return out

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, token_ids: np.ndarray) -> np.ndarray:
        """Token embedding lookup (no positional encoding for Llama).

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(batch, seq)``, int32

        Returns
        -------
        hidden : np.ndarray, shape ``(batch, seq, n_embd)``, float32
        """
        return self.wte[token_ids.ravel()].reshape(
            token_ids.shape[0], token_ids.shape[1], -1
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray, temperature: float, top_k: int) -> int:
        """Sample next token from logits. Returns int."""
        # Use numpy softmax on CPU — avoid GPU roundtrip for sampling
        if temperature > 1e-6:
            scaled = logits / temperature
            # NumPy softmax
            logits_stable = scaled - np.max(scaled)
            probs = np.exp(logits_stable).astype(np.float64)
            probs = probs / np.sum(probs)
            if top_k > 0:
                top_k = min(top_k, len(probs))
                indices = np.argpartition(probs, -top_k)[-top_k:]
                filtered = np.zeros_like(probs)
                filtered[indices] = probs[indices]
                filtered /= filtered.sum()
                probs = filtered
            return int(np.random.choice(len(probs), p=probs))
        else:
            return int(np.argmax(logits))

    def generate_stream(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        """Autoregressive generation yielding tokens one by one.

        Like ``generate()`` but yields each newly generated token ID
        immediately after its decode step finishes, enabling true
        token-by-token streaming.  Yields ``(token_id, step_time_seconds)``
        where *step_time* is the wall time of the decode step (or prefill
        for the first token).

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Prompt token IDs.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature.  Values close to 0 produce greedy
            decoding; values > 0 enable stochastic sampling.
        top_k : int
            If > 0, restrict sampling to the top-k most probable tokens.

        Yields
        ------
        (token_id, step_time)
            token_id : int
                The newly generated token ID.
            step_time : float
                Wall time of the decode step that produced this token.
        """
        if token_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {token_ids.shape[0]}")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must have at least 1 token")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        total_tokens = token_ids.shape[1] + max_new_tokens
        if total_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"total tokens ({total_tokens} = {token_ids.shape[1]} prompt "
                f"+ {max_new_tokens} new) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )

        self._init_cache()

        # Prefill: forward with cache, stores K/V for all positions
        t0 = time.time()
        logits = self.forward(token_ids, use_cache=True)

        # Generate loop
        tokens = token_ids.copy()
        t_decode_start: float = 0.0  # set after first decode forward
        for step in range(max_new_tokens):
            # Sample from current logits
            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_token = self._sample(next_logits, temperature, top_k)
            tokens = np.concatenate(
                [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
            )

            if step == 0:
                # TTFT: prefill → first token sampled
                step_time = time.time() - t0
            else:
                # Per-token latency: time spent in the previous forward()
                step_time = time.time() - t_decode_start

            yield next_token, step_time

            if step < max_new_tokens - 1:
                # Single-token decode step — measure its wall time
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                t_decode_start = time.time()
                logits = self.forward(new_token_arr, use_cache=True)

    def generate(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> np.ndarray:
        """Autoregressive generation with KV cache.

        Prefill -> then incremental decode with cached K/V.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Prompt token IDs.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature.  Values close to 0 produce greedy
            decoding; values > 0 enable stochastic sampling.
        top_k : int
            If > 0, restrict sampling to the top-k most probable tokens.

        Returns
        -------
        out : np.ndarray, shape ``(1, seq + max_new_tokens)``, int32
            Prompt extended with generated tokens.
        """
        if token_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {token_ids.shape[0]}")
        if token_ids.shape[1] == 0:
            raise ValueError("token_ids must have at least 1 token")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        total_tokens = token_ids.shape[1] + max_new_tokens
        if total_tokens > self.config.max_position_embeddings:
            raise ValueError(
                f"total tokens ({total_tokens} = {token_ids.shape[1]} prompt "
                f"+ {max_new_tokens} new) exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )

        self._init_cache()

        # Prefill: forward with cache, stores K/V for all positions
        logits = self.forward(token_ids, use_cache=True)

        # Generate loop
        tokens = token_ids.copy()
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_token = self._sample(next_logits, temperature, top_k)
            tokens = np.concatenate(
                [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
            )
            if step < max_new_tokens - 1:
                # Single-token decode step
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                logits = self.forward(new_token_arr, use_cache=True)
        return tokens
