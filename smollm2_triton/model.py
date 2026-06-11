"""SmolLM2 (Llama-architecture) inference model with KV cache.

Wires RMSNorm, RoPE, SwiGLU, and GQA attention Triton kernels into
a complete autoregressive language model.  Batch=1 only.

GPU-resident path only (EC32 Phase 1).
"""

import time

import numpy as np

from triton_llm import gpu
from triton_llm.kernels.gemm import gemm
from triton_llm.kernels.add import add
from triton_llm.kernels.rms_norm import rms_norm
from triton_llm.kernels.rope import precompute_cos_sin_device, apply_rope
from triton_llm.kernels.swiglu import swiglu
from triton_llm.kernels.attention_gqa import attention_gqa

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

        # --- GPU-resident weight storage ---
        # Lazily initialized by _init_gpu_weights().
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
    # KV cache (GPU-resident path)
    # ------------------------------------------------------------------

    def _init_cache_gpu(self, max_seq: int | None = None):
        """Initialise pre-allocated KV cache for GPU-resident path.

        Allocates host-side numpy arrays for KV cache storage. In Phase 1,
        cache read/write goes through CPU (numpy) due to reshape/transpose
        constraints. Phase 2 will move the KV cache to GPU-resident
        DeviceTensors to eliminate the round-trips.

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
        self.kv_cache_dev = [
            {
                "k": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
                "v": np.zeros((n_kv_head, max_seq, d_k), dtype=np.float32),
            }
            for _ in range(n_layer)
        ]

    # ------------------------------------------------------------------
    # Forward (GPU-resident)
    # ------------------------------------------------------------------

    def _forward_cached(self, token_ids: np.ndarray) -> np.ndarray:
        """GPU-resident incremental forward pass using KV cache.

        Keeps hidden state and most intermediate tensors on GPU.
        KV cache read/write goes through CPU (numpy) due to reshape/transpose
        constraints.  Only one ``synchronize()`` and one ``to_host()`` at the end.

        For decode (seq == 1), QKV gemm output ``(1, n_heads*d_k)`` is already
        head-major contiguous, so the reshape to ``(n_heads, d_k)`` is a zero-copy
        ``gpu.view()`` — no CPU round-trip needed.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd
        n_head = config.n_head
        n_kv_head = config.n_kv_head
        d_k = n_embd // n_head

        self._init_gpu_weights()

        if not hasattr(self, "kv_cache_dev") or self.kv_cache_dev is None:
            raise RuntimeError("_init_cache_gpu() must be called before _forward_cached()")

        prev_seq = self._cache_len
        seq = token_ids.shape[1]

        is_prefill = (prev_seq == 0)
        if not is_prefill and seq > 1:
            raise ValueError(
                f"Decode mode requires seq=1, got seq={seq}. "
            )
        total_after = prev_seq + seq
        max_seq = self.kv_cache_dev[0]["k"].shape[1]
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
            rms_norm(h_dev, self.ln_1_w_dev[i], ln_out_dev, config.rms_norm_eps)

            # QKV projections on GPU
            q_dev = gemm(ln_out_dev, self.q_proj_w_dev[i])  # (seq, n_head * d_k)
            k_dev = gemm(ln_out_dev, self.k_proj_w_dev[i])  # (seq, n_kv_head * d_k)
            v_dev = gemm(ln_out_dev, self.v_proj_w_dev[i])  # (seq, n_kv_head * d_k)

            if seq == 1:
                # --- Decode fast path: zero-copy view, no CPU round-trips ---
                # For seq=1, (1, n_head*d_k) is contiguous head-major = (n_head, d_k).
                q_hm = gpu.view(q_dev, (n_head, d_k))
                k_hm = gpu.view(k_dev, (n_kv_head, d_k))
                v_hm = gpu.view(v_dev, (n_kv_head, d_k))

                apply_rope(q_hm, self.cos_dev, self.sin_dev, seq_len=1, position_offset=prev_seq)
                apply_rope(k_hm, self.cos_dev, self.sin_dev, seq_len=1, position_offset=prev_seq)

                # Cache write (CPU numpy — unavoidable for Phase 1 cache storage)
                assert k_hm.shape == (n_kv_head, d_k), f"k_hm shape {k_hm.shape} != ({n_kv_head}, {d_k})"
                cache["k"][:, prev_seq:prev_seq + 1, :] = k_hm.to_numpy().reshape(n_kv_head, 1, d_k)
                cache["v"][:, prev_seq:prev_seq + 1, :] = v_hm.to_numpy().reshape(n_kv_head, 1, d_k)

                # Cache read for decode attention
                k_view_np = cache["k"][:, :total_after, :].reshape(-1, d_k)
                v_view_np = cache["v"][:, :total_after, :].reshape(-1, d_k)
                k_view_dev = gpu.to_device(np.ascontiguousarray(k_view_np))
                v_view_dev = gpu.to_device(np.ascontiguousarray(v_view_np))

                attn_dev = attention_gqa(
                    q_hm, k_view_dev, v_view_dev,
                    n_head, n_kv_head,
                    causal=False,
                )

                # Attention output (n_head, d_k) → (1, n_head*d_k) — zero-copy view
                attn_sm = gpu.view(attn_dev, (1, n_head * d_k))
            else:
                # --- Prefill path: CPU reshape+transpose needed for seq > 1 ---
                q_np = gpu.to_host(q_dev)
                k_np = gpu.to_host(k_dev)
                v_np = gpu.to_host(v_dev)

                q_flat = np.ascontiguousarray(q_np.reshape(seq, n_head, d_k).transpose(1, 0, 2).reshape(-1, d_k))
                k_flat = np.ascontiguousarray(k_np.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k))
                v_flat = np.ascontiguousarray(v_np.reshape(seq, n_kv_head, d_k).transpose(1, 0, 2).reshape(-1, d_k))

                q_dev = gpu.to_device(q_flat)
                k_dev = gpu.to_device(k_flat)
                v_dev = gpu.to_device(v_flat)

                apply_rope(q_dev, self.cos_dev, self.sin_dev, seq_len=seq, position_offset=prev_seq)
                apply_rope(k_dev, self.cos_dev, self.sin_dev, seq_len=seq, position_offset=prev_seq)

                # Always is_prefill here (seq>1 decode is handled by fast path above)
                k_np_cache = gpu.to_host(k_dev).reshape(n_kv_head, seq, d_k)
                v_np_cache = gpu.to_host(v_dev).reshape(n_kv_head, seq, d_k)
                cache["k"][:, :seq, :] = k_np_cache
                cache["v"][:, :seq, :] = v_np_cache

                attn_dev = attention_gqa(
                    q_dev, k_dev, v_dev, n_head, n_kv_head, causal=True
                )

                # Transpose attention output back: (n_head*seq, d_k) -> (seq, n_head*d_k)
                attn_np = gpu.to_host(attn_dev)
                attn_seq_major = np.ascontiguousarray(attn_np.reshape(n_head, seq, d_k).transpose(1, 0, 2).reshape(seq, n_head * d_k))
                attn_sm = gpu.to_device(attn_seq_major)

            # Output projection + residual on GPU
            o_dev = gemm(attn_sm, self.o_proj_w_dev[i])  # (seq, n_embd)
            h_dev = gpu.allocate((seq, n_embd), np.float32)
            add(o_dev, residual_dev, out_dev=h_dev)

            # --- MLP sub-block ---
            residual_dev = h_dev
            ln_out_dev = gpu.allocate((seq, n_embd), np.float32)
            rms_norm(h_dev, self.ln_2_w_dev[i], ln_out_dev, config.rms_norm_eps)

            gate_dev = gemm(ln_out_dev, self.gate_proj_w_dev[i])  # (seq, intermediate_size)
            up_dev = gemm(ln_out_dev, self.up_proj_w_dev[i])      # (seq, intermediate_size)
            act_dev = swiglu(gate_dev, up_dev)                     # (seq, intermediate_size)
            down_dev = gemm(act_dev, self.down_proj_w_dev[i])      # (seq, n_embd)
            h_dev = gpu.allocate((seq, n_embd), np.float32)
            add(down_dev, residual_dev, out_dev=h_dev)

        # Update cache length after processing this step
        self._cache_len = total_after

        # --- Final RMSNorm on GPU ---
        ln_out_dev = gpu.allocate((seq, n_embd), np.float32)
        rms_norm(h_dev, self.ln_f_w_dev, ln_out_dev, config.rms_norm_eps)

        # --- LM head on GPU ---
        logits_dev = gemm(ln_out_dev, self.lm_head_w_dev)  # (seq, vocab_size)

        # Single sync + to_host at the very end
        gpu.synchronize()
        return gpu.to_host(logits_dev).reshape(1, seq, config.vocab_size)

    def forward_gpu(self, token_ids: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """GPU-resident forward pass.

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
            if not hasattr(self, "kv_cache_dev") or self.kv_cache_dev is None:
                raise RuntimeError(
                    "_init_cache_gpu() must be called before forward_gpu(use_cache=True)"
                )
        else:
            self._init_cache_gpu()
        return self._forward_cached(token_ids)

    def generate_gpu(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
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
        logits = self._forward_cached(token_ids)

        tokens = token_ids.copy()
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :]
            next_token = self._sample(next_logits, temperature, top_k)
            tokens = np.concatenate(
                [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
            )
            if next_token == eos_token_id:
                break
            if step < max_new_tokens - 1:
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                logits = self._forward_cached(new_token_arr)
        return tokens

    def generate_stream_gpu(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
    ):
        """GPU-resident generation yielding tokens one by one with real timing.

        Like ``generate_stream()`` but uses the GPU-resident forward pass.
        Yields ``(token_id, step_time_seconds)`` where *step_time* is the
        wall time of the prefill (first token) or decode step (subsequent).

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

        Yields
        ------
        (token_id, step_time)
            token_id : int
            step_time : float — wall time of prefill (first yield) or decode step.
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
        t0 = time.perf_counter()
        logits = self._forward_cached(token_ids)

        t_decode_start: float = 0.0
        for step in range(max_new_tokens):
            next_logits = logits[0, -1, :]
            next_token = self._sample(next_logits, temperature, top_k)

            if step == 0:
                step_time = time.perf_counter() - t0  # TTFT: prefill → first token
            else:
                step_time = time.perf_counter() - t_decode_start  # TPOT

            yield next_token, step_time

            if eos_token_id is not None and next_token == eos_token_id:
                return
            if step < max_new_tokens - 1:
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                t_decode_start = time.perf_counter()
                logits = self._forward_cached(new_token_arr)

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
    # Sampling
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

    # ------------------------------------------------------------------
    # Backward-compatibility aliases
    # ------------------------------------------------------------------

    def _init_cache(self, max_seq: int | None = None):
        """Alias for ``_init_cache_gpu``. Provided for backward compatibility."""
        return self._init_cache_gpu(max_seq)

    def forward(self, token_ids: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """Alias for ``forward_gpu``. Provided for backward compatibility."""
        return self.forward_gpu(token_ids, use_cache)

    def generate(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> np.ndarray:
        """Alias for ``generate_gpu``. Provided for backward compatibility."""
        return self.generate_gpu(token_ids, max_new_tokens, temperature, top_k)

    def generate_stream(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ):
        """Alias for ``generate_stream_gpu``. Provided for backward compatibility."""
        yield from self.generate_stream_gpu(token_ids, max_new_tokens, temperature, top_k)
