"""
GPT-2 Model with forward and generate using Triton kernels.

Supports KV-cached incremental decode for efficient autoregressive generation.
Batch=1 only.
"""

import numpy as np

from .config import GPT2Config
from .kernels.add import add
from .kernels.attention import attention
from .kernels.embedding import embedding
from .kernels.gelu import gelu
from .kernels.gemm import gemm
from .kernels.layernorm import layer_norm
from .kernels.softmax import softmax


class GPT2Model:
    """GPT-2 inference model backed by Triton kernels.

    Parameters
    ----------
    config : GPT2Config
        Model configuration.
    weights : dict of str -> np.ndarray
        Dictionary mapping HF weight names to NumPy arrays.
        Weight matrices are transposed from HF's (out, in) layout
        to (in, out) at load time.
    """

    def __init__(self, config: GPT2Config, weights: dict):
        self.config = config
        n_layer = config.n_layer

        # --- Embedding tables (stored as-is, (vocab, n_embd) and (pos, n_embd)) ---
        # Wte also serves as lm_head (tied embeddings).
        # Pre-transpose for gemm(hidden, wte.T): store (n_embd, vocab_size) C-contiguous.
        self.wte = np.require(
            weights["wte.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.lm_head_w = np.require(
            self.wte.T, dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.wpe = np.require(
            weights["wpe.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )

        # --- Final layer norm ---
        self.ln_f_g = np.require(
            weights["ln_f.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.ln_f_b = np.require(
            weights["ln_f.bias"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )

        # --- Per-layer weight lists ---
        self.ln_1_g: list[np.ndarray] = []
        self.ln_1_b: list[np.ndarray] = []
        self.ln_2_g: list[np.ndarray] = []
        self.ln_2_b: list[np.ndarray] = []
        self.c_attn_w: list[np.ndarray] = []
        self.c_attn_b: list[np.ndarray] = []
        self.c_proj_w: list[np.ndarray] = []
        self.c_proj_b: list[np.ndarray] = []
        self.c_fc_w: list[np.ndarray] = []
        self.c_fc_b: list[np.ndarray] = []
        self.c_proj_mlp_w: list[np.ndarray] = []
        self.c_proj_mlp_b: list[np.ndarray] = []

        for i in range(n_layer):
            # --- Attention block ---
            # GPT-2 uses Conv1D layers: weights are stored as (in_features, out_features).
            # No transpose needed — gemm(hidden, weight) expects (seq, n_embd) @ (n_embd, out).
            w = weights[f"h.{i}.attn.c_attn.weight"]
            self.c_attn_w.append(
                np.require(w, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_attn_b.append(
                np.require(
                    weights[f"h.{i}.attn.c_attn.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # c_proj: HF (n_embd, n_embd) — already (in, out), no transpose
            w = weights[f"h.{i}.attn.c_proj.weight"]
            self.c_proj_w.append(
                np.require(w, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_proj_b.append(
                np.require(
                    weights[f"h.{i}.attn.c_proj.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # --- MLP block ---
            # c_fc: HF (n_embd, 4*n_embd) — already (in, out), no transpose
            w = weights[f"h.{i}.mlp.c_fc.weight"]
            self.c_fc_w.append(
                np.require(w, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_fc_b.append(
                np.require(
                    weights[f"h.{i}.mlp.c_fc.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # c_proj (MLP output): HF (4*n_embd, n_embd) — already (in, out), no transpose
            w = weights[f"h.{i}.mlp.c_proj.weight"]
            self.c_proj_mlp_w.append(
                np.require(w, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_proj_mlp_b.append(
                np.require(
                    weights[f"h.{i}.mlp.c_proj.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # --- Layer norms ---
            self.ln_1_g.append(
                np.require(
                    weights[f"h.{i}.ln_1.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_1_b.append(
                np.require(
                    weights[f"h.{i}.ln_1.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_2_g.append(
                np.require(
                    weights[f"h.{i}.ln_2.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_2_b.append(
                np.require(
                    weights[f"h.{i}.ln_2.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """Run a forward pass over input tokens.

        When ``use_cache=False`` (default), this performs a full forward pass
        over all input tokens (no KV cache).

        When ``use_cache=True``, this uses the KV cache for incremental
        decode.  On the first call, the cache is populated (prefill); on
        subsequent calls, only the last token is processed using cached
        K/V from previous steps.

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
        if use_cache:
            return self._forward_cached(token_ids)
        return self._forward_full(token_ids)

    def _forward_full(self, token_ids: np.ndarray) -> np.ndarray:
        """Run a full forward pass over *all* input tokens (no KV cache).

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Input token IDs.

        Returns
        -------
        logits : np.ndarray, shape ``(1, seq, vocab_size)``, float32
            Unnormalised logits for each position.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd

        # --- Embedding ---
        hidden = embedding(token_ids, self.wte, self.wpe)  # (1, seq, n_embd)

        # --- Transformer layers ---
        for i in range(n_layer):
            # --- Attention sub-block ---
            residual = hidden
            h = hidden.reshape(-1, n_embd)                     # (seq, n_embd)
            h = layer_norm(h, self.ln_1_g[i], self.ln_1_b[i],
                           config.layer_norm_epsilon)
            h = self._apply_attention(h, i)                     # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)

            # --- MLP sub-block ---
            residual = hidden
            h = hidden.reshape(-1, n_embd)                     # (seq, n_embd)
            h = layer_norm(h, self.ln_2_g[i], self.ln_2_b[i],
                           config.layer_norm_epsilon)
            h = self._apply_mlp(h, i)                           # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)

        # --- Final layer norm ---
        h = hidden.reshape(-1, n_embd)                          # (seq, n_embd)
        h = layer_norm(h, self.ln_f_g, self.ln_f_b,
                       config.layer_norm_epsilon)

        # --- LM head ---
        logits = gemm(h, self.lm_head_w)  # (seq, n_embd) @ (n_embd, vocab) -> (seq, vocab)
        return logits.reshape(1, -1, config.vocab_size)

    # ------------------------------------------------------------------
    # Attention sub-block
    # ------------------------------------------------------------------

    def _init_cache(self):
        """Initialize empty KV cache for a new generation."""
        n_layer = self.config.n_layer
        n_head = self.config.n_head
        n_embd = self.config.n_embd
        d_k = n_embd // n_head
        # Per layer: list of {'k': [Head x np.array(cached_seq, d_k)], 'v': [Head x ...]}
        self.kv_cache = [
            {
                "k": [np.empty((0, d_k), dtype=np.float32) for _ in range(n_head)],
                "v": [np.empty((0, d_k), dtype=np.float32) for _ in range(n_head)],
            }
            for _ in range(n_layer)
        ]

    def _forward_cached(self, token_ids: np.ndarray) -> np.ndarray:
        """Incremental forward pass using KV cache.

        Supports two modes:
        - Prefill (self.kv_cache is empty): process all tokens, store K/V for
          all layers/heads.
        - Decode (self.kv_cache has data): process only the last token using
          cached K/V from previous steps.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Input token IDs.  For decode steps, seq=1.

        Returns
        -------
        logits : np.ndarray, shape ``(1, seq, vocab_size)``, float32
            Unnormalised logits for each position.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd
        n_head = config.n_head
        d_k = n_embd // n_head

        # Compute position offset: total tokens processed before this call
        # (including all prior prompt + generated tokens).
        # Use the first head's first-layer cache length as the running count.
        prev_seq = self.kv_cache[0]["k"][0].shape[0]
        pos_offset = prev_seq

        # Embedding with position offset for correct positional encoding
        hidden = embedding(token_ids, self.wte, self.wpe,
                           position_offset=pos_offset)  # (1, seq, n_embd)
        seq = hidden.shape[1]
        hidden = hidden.reshape(-1, n_embd)  # (seq, n_embd)

        for i in range(n_layer):
            # --- Attention sub-block ---
            residual = hidden  # (seq, n_embd)
            h = layer_norm(hidden, self.ln_1_g[i], self.ln_1_b[i],
                           config.layer_norm_epsilon)  # (seq, n_embd)

            # QKV projection
            qkv = gemm(h, self.c_attn_w[i])  # (seq, 3*n_embd)
            qkv = add(qkv, np.broadcast_to(self.c_attn_b[i], qkv.shape))
            q_all = qkv[:, :n_embd]   # (seq, n_embd)
            k_all = qkv[:, n_embd:2 * n_embd]  # (seq, n_embd)
            v_all = qkv[:, 2 * n_embd:]  # (seq, n_embd)

            cache = self.kv_cache[i]
            attn_out = np.empty(hidden.shape, dtype=np.float32)  # (seq, n_embd)

            for head in range(n_head):
                s = head * d_k
                e = s + d_k
                q_h = q_all[:, s:e]   # (seq, d_k)
                k_h = k_all[:, s:e]   # (seq, d_k)
                v_h = v_all[:, s:e]   # (seq, d_k)

                # Append to cache
                cache["k"][head] = np.concatenate([cache["k"][head], k_h], axis=0)
                cache["v"][head] = np.concatenate([cache["v"][head], v_h], axis=0)

                # Fused attention -- with or without causal masking
                # Prefill: K/V seq equals Q seq => causal=True
                # Decode: K/V seq > Q seq => causal=False
                # NOTE: This check relies on the np.concatenate above
                # having already appended k_h/k_v. If the append is ever
                # moved after the attention call, this logic breaks.
                is_prefill = (cache["k"][head].shape[0] == seq)
                o_h = attention(q_h, cache["k"][head], cache["v"][head],
                                causal=is_prefill)
                attn_out[:, s:e] = o_h

            # Output projection
            out = gemm(attn_out, self.c_proj_w[i])  # (seq, n_embd)
            out = add(out, np.broadcast_to(self.c_proj_b[i], out.shape))
            hidden = add(out, residual)  # (seq, n_embd)

            # --- MLP sub-block (unchanged, no cache) ---
            residual = hidden
            h = layer_norm(hidden, self.ln_2_g[i], self.ln_2_b[i],
                           config.layer_norm_epsilon)
            h = self._apply_mlp(h, i)
            hidden = add(h, residual)

        # Final LN
        h = layer_norm(hidden, self.ln_f_g, self.ln_f_b,
                       config.layer_norm_epsilon)
        # LM head
        logits = gemm(h, self.lm_head_w)  # (seq, vocab_size)
        return logits.reshape(1, -1, config.vocab_size)

    # ------------------------------------------------------------------
    # Attention sub-block
    # ------------------------------------------------------------------

    def _apply_attention(self, h: np.ndarray, layer_idx: int) -> np.ndarray:
        """QKV projection -> split -> per-head causal attention -> output proj."""
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        d_k = n_embd // n_head

        # QKV projection: (seq, n_embd) -> (seq, 3*n_embd)
        qkv = gemm(h, self.c_attn_w[layer_idx])  # (seq, 3*n_embd)
        qkv = add(qkv, np.broadcast_to(self.c_attn_b[layer_idx], qkv.shape))

        # Split into Q, K, V (each (seq, n_embd))
        q_all = qkv[:, :n_embd]
        k_all = qkv[:, n_embd:2 * n_embd]
        v_all = qkv[:, 2 * n_embd:]

        seq = h.shape[0]
        attn_out = np.empty((seq, n_embd), dtype=np.float32)

        # Per-head causal attention
        for head in range(n_head):
            s = head * d_k
            e = s + d_k
            q_h = q_all[:, s:e]   # (seq, d_k)
            k_h = k_all[:, s:e]   # (seq, d_k)
            v_h = v_all[:, s:e]   # (seq, d_k)
            o_h = attention(q_h, k_h, v_h)  # (seq, d_k)
            attn_out[:, s:e] = o_h

        # Output projection: (seq, n_embd) -> (seq, n_embd)
        out = gemm(attn_out, self.c_proj_w[layer_idx])  # (seq, n_embd)
        return add(out, np.broadcast_to(self.c_proj_b[layer_idx], out.shape))

    # ------------------------------------------------------------------
    # MLP sub-block
    # ------------------------------------------------------------------

    def _apply_mlp(self, h: np.ndarray, layer_idx: int) -> np.ndarray:
        """FC up -> GELU -> FC down."""
        # (seq, n_embd) -> (seq, 4*n_embd)
        h = gemm(h, self.c_fc_w[layer_idx])  # (seq, 4*n_embd)
        h = add(h, np.broadcast_to(self.c_fc_b[layer_idx], h.shape))
        h = gelu(h)  # (seq, 4*n_embd)
        # (seq, 4*n_embd) -> (seq, n_embd)
        out = gemm(h, self.c_proj_mlp_w[layer_idx])  # (seq, n_embd)
        return add(out, np.broadcast_to(self.c_proj_mlp_b[layer_idx], out.shape))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _sample(self, logits: np.ndarray, temperature: float, top_k: int) -> int:
        """Sample next token from logits. Returns int."""
        if temperature > 1e-6:
            scaled = logits / temperature
            probs = softmax(scaled.reshape(1, -1)).ravel()
            if top_k > 0:
                indices = np.argpartition(probs, -top_k)[-top_k:]
                filtered = np.zeros_like(probs)
                filtered[indices] = probs[indices]
                filtered /= filtered.sum()
                probs = filtered
            return int(np.random.choice(len(probs), p=probs))
        else:
            return int(np.argmax(logits))

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

        self._init_cache()

        # Prefill: forward with cache, stores K/V for all positions
        logits = self.forward(token_ids, use_cache=True)

        # Generate loop
        tokens = token_ids.copy()
        for _ in range(max_new_tokens):
            next_logits = logits[0, -1, :]  # (vocab_size,)
            next_token = self._sample(next_logits, temperature, top_k)
            tokens = np.concatenate(
                [tokens, np.array([[next_token]], dtype=np.int32)], axis=1
            )
            if _ < max_new_tokens - 1:
                # Single-token decode step
                new_token_arr = np.array([[next_token]], dtype=np.int32)
                logits = self.forward(new_token_arr, use_cache=True)
        return tokens
